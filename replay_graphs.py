#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dump out stats about all the actions that are in use in a set of replays."""

import collections
import multiprocessing
import os
import queue
import signal
import sys
import threading
import time
import io
import json
import mpyq
import numpy as np
import pickle

from absl import app
from absl import flags

from pysc2 import run_configs
from pysc2.lib import features
from pysc2.lib import point
from pysc2.lib import protocol
from pysc2.lib import remote_controller
from pysc2.lib import replay
from pysc2.lib import static_data

from pysc2.lib import gfile
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.run_configs import lib as run_configs_lib

from sc2_graph import get_deltas, check_battle, create_graph
from graph_image import create_graph_image

NUM_ACTION_TYPES = 539
MAX_UNIT_COUNT = 16
NUM_UNIT_TYPES = 243
NUM_UNIT_FEATURES = 40
NUM_UPGRADES = 40
NUM_UPGRADE_TYPES = 86
MAX_UNIT_SELECTION_SIZE = 16
MAP_SIZE = 128
RAW_RESOLUTION = 128
MINIMAP_SIZE = 64
SCREEN_SIZE = 96

FLAGS = flags.FLAGS
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_integer("step_mul", 8, "How many game steps per observation.")
flags.DEFINE_string("replays", "/home/ahedzic/StarCraftII/Replays", "Path to a directory of replays.")
flags.mark_flag_as_required("replays")
graph_dir ='/home/ahedzic/Downloads/graphs'
graph_images_dir ='/home/ahedzic/Downloads/graph_images'


size = point.Point(16, 16)
interface = sc_pb.InterfaceOptions(
    raw=True, score=False,
    feature_layer=sc_pb.SpatialCameraSetup(width=24))
size.assign_to(interface.feature_layer.resolution)
size.assign_to(interface.feature_layer.minimap_resolution)

def _make_converter_settings_common(**kwargs):
  return converter_pb2.ConverterSettings(
      num_action_types=NUM_ACTION_TYPES,
      num_unit_types=NUM_UNIT_TYPES,
      num_upgrade_types=NUM_UPGRADE_TYPES,
      max_num_upgrades=NUM_UPGRADES,
      minimap=common_pb2.Size2DI(x=MINIMAP_SIZE, y=MINIMAP_SIZE),
      minimap_features=['height_map', 'visibility_map'],
      add_opponent_features=True,
      **kwargs)


def _make_converter_settings(mode: str):
  if mode == 'visual':
    return _make_converter_settings_common(
        visual_settings=converter_pb2.ConverterSettings.VisualSettings(
            screen=common_pb2.Size2DI(x=SCREEN_SIZE, y=SCREEN_SIZE),
            screen_features=['height_map', 'player_relative']))
  else:
    return _make_converter_settings_common(
        raw_settings=converter_pb2.ConverterSettings.RawSettings(
            resolution=common_pb2.Size2DI(x=RAW_RESOLUTION, y=RAW_RESOLUTION),
            num_unit_features=NUM_UNIT_FEATURES,
            max_unit_count=MAX_UNIT_COUNT,
            max_unit_selection_size=MAX_UNIT_SELECTION_SIZE,
            enable_action_repeat=True))

def get_replay_version(replay_data):
  replay_io = io.BytesIO()
  replay_io.write(replay_data)
  replay_io.seek(0)
  archive = mpyq.MPQArchive(replay_io).extract()
  metadata = json.loads(archive[b"replay.gamemetadata.json"].decode("utf-8"))
  return run_configs_lib.Version(
      game_version=".".join(metadata["GameVersion"].split(".")[:-1]),
      build_version=int(metadata["BaseBuild"][4:]),
      data_version=metadata.get("DataVersion"),  # Only in replays version 4.1+.
      binary=None)


def sorted_dict_str(d):
  return "{%s}" % ", ".join("%s: %s" % (k, d[k])
                            for k in sorted(d, key=d.get, reverse=True))


class ReplayStats(object):
  """Summary stats of the replays seen so far."""

  def __init__(self):
    self.replays = 0
    self.steps = 0
    self.camera_move = 0
    self.select_pt = 0
    self.select_rect = 0
    self.control_group = 0
    self.maps = collections.defaultdict(int)
    self.races = collections.defaultdict(int)
    self.unit_ids = collections.defaultdict(int)
    self.valid_abilities = collections.defaultdict(int)
    self.made_abilities = collections.defaultdict(int)
    self.valid_actions = collections.defaultdict(int)
    self.made_actions = collections.defaultdict(int)
    self.buffs = collections.defaultdict(int)
    self.upgrades = collections.defaultdict(int)
    self.effects = collections.defaultdict(int)
    self.crashing_replays = set()
    self.invalid_replays = set()

  def merge(self, other):
    """Merge another ReplayStats into this one."""
    def merge_dict(a, b):
      for k, v in b.items():
        a[k] += v

    self.replays += other.replays
    self.steps += other.steps
    self.camera_move += other.camera_move
    self.select_pt += other.select_pt
    self.select_rect += other.select_rect
    self.control_group += other.control_group
    merge_dict(self.maps, other.maps)
    merge_dict(self.races, other.races)
    merge_dict(self.unit_ids, other.unit_ids)
    merge_dict(self.valid_abilities, other.valid_abilities)
    merge_dict(self.made_abilities, other.made_abilities)
    merge_dict(self.valid_actions, other.valid_actions)
    merge_dict(self.made_actions, other.made_actions)
    merge_dict(self.buffs, other.buffs)
    merge_dict(self.upgrades, other.upgrades)
    merge_dict(self.effects, other.effects)
    self.crashing_replays |= other.crashing_replays
    self.invalid_replays |= other.invalid_replays

  def __str__(self):
    len_sorted_dict = lambda s: (len(s), sorted_dict_str(s))
    len_sorted_list = lambda s: (len(s), sorted(s))

    new_abilities = ((set(self.valid_abilities.keys())
                      | set(self.made_abilities.keys()))
                     - set(static_data.ABILITIES))
    new_units = set(self.unit_ids) - set(static_data.UNIT_TYPES)
    new_buffs = set(self.buffs) - set(static_data.BUFFS)
    new_upgrades = set(self.upgrades) - set(static_data.UPGRADES)
    return "\n\n".join((
        "Replays: %s, Steps total: %s" % (self.replays, self.steps),
        "Camera move: %s, Select pt: %s, Select rect: %s, Control group: %s" % (
            self.camera_move, self.select_pt, self.select_rect,
            self.control_group),
        "Maps: %s\n%s" % len_sorted_dict(self.maps),
        "Races: %s\n%s" % len_sorted_dict(self.races),
        #"Unit ids: %s\n%s" % len_sorted_dict(self.unit_ids),
        #"New units: %s \n%s" % len_sorted_list(new_units),
        #"Valid abilities: %s\n%s" % len_sorted_dict(self.valid_abilities),
        #"Made abilities: %s\n%s" % len_sorted_dict(self.made_abilities),
        #"New abilities: %s\n%s" % len_sorted_list(new_abilities),
        #"Valid actions: %s\n%s" % len_sorted_dict(self.valid_actions),
        #"Made actions: %s\n%s" % len_sorted_dict(self.made_actions),
        #"Buffs: %s\n%s" % len_sorted_dict(self.buffs),
        #"New buffs: %s\n%s" % len_sorted_list(new_buffs),
        #"Upgrades: %s\n%s" % len_sorted_dict(self.upgrades),
        #"New upgrades: %s\n%s" % len_sorted_list(new_upgrades),
        #"Effects: %s\n%s" % len_sorted_dict(self.effects),
        "Crashing replays: %s\n%s" % len_sorted_list(self.crashing_replays),
        "Invalid replays: %s\n%s" % len_sorted_list(self.invalid_replays),
    ))


class ProcessStats(object):
  """Stats for a worker process."""

  def __init__(self, proc_id):
    self.proc_id = proc_id
    self.time = time.time()
    self.stage = ""
    self.replay = ""
    self.replay_stats = ReplayStats()

  def update(self, stage):
    self.time = time.time()
    self.stage = stage

  def __str__(self):
    return ("[%2d] replay: %10s, replays: %5d, steps: %7d, game loops: %7s, "
            "last: %12s, %3d s ago" % (
                self.proc_id, self.replay, self.replay_stats.replays,
                self.replay_stats.steps,
                self.replay_stats.steps * FLAGS.step_mul, self.stage,
                time.time() - self.time))


def valid_replay(info, ping):
  """Make sure the replay isn't corrupt, and is worth looking at."""
  if (info.HasField("error") or
      info.base_build != ping.base_build or  # different game version
      info.game_duration_loops < 1000 or
      len(info.player_info) != 2):
    # Probably corrupt, or just not interesting.
    return False
  for p in info.player_info:
    if p.player_apm < 10 or p.player_mmr < 1000:
      # Low APM = player just standing around.
      # Low MMR = corrupt replay or player who is weak.
      return False
  return True


class ReplayProcessor(multiprocessing.Process):
  """A Process that pulls replays and processes them."""

  def __init__(self, proc_id, run_config, replay_queue, stats_queue):
    super(ReplayProcessor, self).__init__()
    self.stats = ProcessStats(proc_id)
    self.run_config = run_config
    self.replay_queue = replay_queue
    self.stats_queue = stats_queue

  def run(self):
    signal.signal(signal.SIGTERM, lambda a, b: sys.exit())  # Exit quietly.
    self._update_stage("spawn")
    replay_name = "none"
    while True:
      self._print("Starting up a new SC2 instance.")
      self._update_stage("launch")
      try:
        with self.run_config.start(
            want_rgb=interface.HasField("render")) as controller:
          self._print("SC2 Started successfully.")
          ping = controller.ping()
          for _ in range(300):
            try:
              replay_path = self.replay_queue.get()
            except queue.Empty:
              self._update_stage("done")
              self._print("Empty queue, returning")
              return
            try:
              replay_name = os.path.basename(replay_path)[:10]
              self.stats.replay = replay_name
              self._print("Got replay: %s" % replay_path)
              self._update_stage("open replay file")
              replay_data = self.run_config.replay_data(replay_path)
              self._update_stage("replay_info")
              info = controller.replay_info(replay_data)
              self._print((" Replay Info %s " % replay_name).center(60, "-"))
              self._print(info)
              self._print("-" * 60)
              if valid_replay(info, ping):
                self.stats.replay_stats.maps[info.map_name] += 1
                for player_info in info.player_info:
                  race_name = sc_common.Race.Name(
                      player_info.player_info.race_actual)
                  self.stats.replay_stats.races[race_name] += 1
                map_data = None
                if info.local_map_path:
                  self._update_stage("open map file")
                  map_data = self.run_config.map_data(info.local_map_path)
                for player_id in [1, 2]:
                  self._print("Starting %s from player %s's perspective" % (
                      replay_name, player_id))
                  self.process_replay(controller, replay_data, map_data,
                                      player_id, replay_name)
              else:
                self._print("Replay is invalid.")
                self.stats.replay_stats.invalid_replays.add(replay_name)
            finally:
              self.replay_queue.task_done()
          self._update_stage("shutdown")
      except (protocol.ConnectionError, protocol.ProtocolError,
              remote_controller.RequestError):
        self.stats.replay_stats.crashing_replays.add(replay_name)
      except KeyboardInterrupt:
        return

  def _print(self, s):
    for line in str(s).strip().splitlines():
      print("[%s] %s" % (self.stats.proc_id, line))

  def _update_stage(self, stage):
    self.stats.update(stage)
    self.stats_queue.put(self.stats)
    
  def process_replay(self, controller, replay_data, map_data, player_id, replay_name):
    """Process a single replay, updating the stats."""
    self._update_stage("start_replay")
    controller.start_replay(sc_pb.RequestStartReplay(
        replay_data=replay_data,
        map_data=map_data,
        options=interface,
        observed_player_id=player_id))

    feat = features.features_from_game_info(controller.game_info())

    self.stats.replay_stats.replays += 1
    self._update_stage("step")
    controller.step()
    prev_obs = None
    battle = 1
    battle_step = 1
    current_battle = False
    
    while True:
      self.stats.replay_stats.steps += 1
      self._update_stage("observe")
      obs = controller.observe()
      	
      if (prev_obs != None):
        deltas = get_deltas(obs.observation.raw_data.units, prev_obs.observation.raw_data.units)
        
       	if check_battle(obs.observation.raw_data.units, deltas):
       	  current_battle = True
       	  graph_dict = {}
       	  graph = create_graph(obs.observation.raw_data.units, deltas)
       	  graph_name = replay_name + "_battle_" + str(battle) + "_step_" + str(battle_step)
       	  #create_graph_image(graph_images_dir, graph, graph_name)
       	  graph_path = os.path.join(graph_dir, graph_name+".pkl")
          output = open(graph_path, 'wb')
          pickle.dump(graph_dict, output)
          output.close()
          battle_step += 1
        else:
          if current_battle:
            battle += 1
            current_battle = False
            battle_step = 1
       	  
      prev_obs = obs

      if obs.player_result:
        break

      self._update_stage("step")
      controller.step(FLAGS.step_mul)


def stats_printer(stats_queue):
  """A thread that consumes stats_queue and prints them every 10 seconds."""
  proc_stats = [ProcessStats(i) for i in range(FLAGS.parallel)]
  print_time = start_time = time.time()
  width = 107

  running = True
  while running:
    print_time += 10

    while time.time() < print_time:
      try:
        s = stats_queue.get(True, print_time - time.time())
        if s is None:  # Signal to print and exit NOW!
          running = False
          break
        proc_stats[s.proc_id] = s
      except queue.Empty:
        pass

    replay_stats = ReplayStats()
    for s in proc_stats:
      replay_stats.merge(s.replay_stats)

    print((" Summary %0d secs " % (print_time - start_time)).center(width, "="))
    print(replay_stats)
    print(" Process stats ".center(width, "-"))
    print("\n".join(str(s) for s in proc_stats))
    print("=" * width)


def replay_queue_filler(replay_queue, replay_list):
  """A thread that fills the replay_queue with replay filenames."""
  for replay_path in replay_list:
    replay_queue.put(replay_path)


def main(unused_argv):
  """Dump stats about all the actions that are in use in a set of replays."""
  run_config = run_configs.get()

  if not gfile.Exists(FLAGS.replays):
    sys.exit("{} doesn't exist.".format(FLAGS.replays))

  stats_queue = multiprocessing.Queue()
  stats_thread = threading.Thread(target=stats_printer, args=(stats_queue,))
  try:
    # For some reason buffering everything into a JoinableQueue makes the
    # program not exit, so save it into a list then slowly fill it into the
    # queue in a separate thread. Grab the list synchronously so we know there
    # is work in the queue before the SC2 processes actually run, otherwise
    # The replay_queue.join below succeeds without doing any work, and exits.
    print("Getting replay list:", FLAGS.replays)
    replay_list = sorted(run_config.replay_paths(FLAGS.replays))
    print(len(replay_list), "replays found.")
    if not replay_list:
      return

    if not FLAGS["sc2_version"].present:  # ie not set explicitly.
      version = get_replay_version(
          run_config.replay_data(replay_list[0]))
      run_config = run_configs.get(version=version)
      print("Assuming version:", version.game_version)

    print()

    stats_thread.start()
    replay_queue = multiprocessing.JoinableQueue(FLAGS.parallel * 10)
    replay_queue_thread = threading.Thread(target=replay_queue_filler,
                                           args=(replay_queue, replay_list))
    replay_queue_thread.daemon = True
    replay_queue_thread.start()

    for i in range(min(len(replay_list), FLAGS.parallel)):
      p = ReplayProcessor(i, run_config, replay_queue, stats_queue)
      p.daemon = True
      p.start()
      time.sleep(1)  # Stagger startups, otherwise they seem to conflict somehow

    replay_queue.join()  # Wait for the queue to empty.
  except KeyboardInterrupt:
    print("Caught KeyboardInterrupt, exiting.")
  finally:
    stats_queue.put(None)  # Tell the stats_thread to print and exit.
    if stats_thread.is_alive():
      stats_thread.join()


if __name__ == "__main__":
  app.run(main)

