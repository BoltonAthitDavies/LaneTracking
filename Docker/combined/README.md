# Combined container: LaneTracking + wil_project

One image, one container, both repositories. LaneTracking drives the robot
(CAN, cameras, lane detection, odometry); wil_project provides the VSLAM and
navigation stack (VINS-Fusion, ORB-SLAM3, RTAB-Map, Nav2).

```bash
./docker.sh build     # once, hours
./docker.sh run       # start, drops you in a shell
ws_build              # inside: build wil_project, then LaneTracking
```

`./docker.sh exec` opens a second shell in the running container.

---

## Why this exists instead of combining the two Dockerfiles

The two repositories target different operating systems:

| | LaneTracking | wil_project |
| :-- | :-- | :-- |
| base image | `dustynv/ros:humble-desktop-pytorch-l4t-r35.4.1` | `osrf/ros:humble-desktop` |
| distro | Ubuntu **20.04** (focal) | Ubuntu **22.04** (jammy) |
| ROS 2 Humble | **built from source**, `/opt/ros/humble/install` | apt, `/opt/ros/humble` |
| torch | 2.0.0+nv23.05, L4T CUDA 11.4 | 2.5.1+cu121 from the PyPI CUDA index |
| OpenCV | 4.5.0 **with CUDA**, custom debs in `/lib` | apt 4.5.4 |

On this Jetson (aarch64, L4T R35) wil_project's image cannot be built at all:
there is no aarch64 `cu121` wheel, and JetPack 5 ships no CUDA torch for 22.04.
Its `ros-humble-*` apt dependencies are jammy-only, so on focal they have to
come from source. The L4T base is therefore the only possible base, and
wil_project is ported onto it. Nothing is lost: ROS 2 was never installed on
the host and wil_project had never been built here.

## What is in the image

Already supplied by the L4T base — `cv_bridge`, `image_transport` and plugins,
`image_geometry`, `stereo_image_proc`, `stereo_msgs`, `laser_geometry`,
`pcl_conversions`, PCL 1.10, `tf2_geometry_msgs`, `tf2_sensor_msgs`,
`vision_msgs`, `message_filters`, `angles`, `robot_state_publisher`, `rviz2`,
OpenCV 4.5.0 with CUDA, torch 2.0.0+nv, CUDA 11.4.

Built from source by this Dockerfile:

| | Version | Prefix |
| :-- | :-- | :-- |
| Ceres | 2.2.0 | `/usr/local` |
| g2o | 20201223 | `/opt/wil/g2o` |
| RTAB-Map core | 0.23.7 | `/opt/wil/rtabmap` |
| `bondcpp`, `diagnostic_updater`, `pcl_ros`, `behaviortree_cpp_v3` 3.8, `joint_state_publisher`, `rosbag2_storage_mcap` | humble | `/opt/wil/rosdeps` |
| Nav2 (subset) | humble | `/opt/wil/nav2` |
| ORB-SLAM3 (fork `5c2b00f`) + Pangolin 0.9.1 | pinned | `/opt/wil/thirdparty` |

Neither repository is copied into the image. `docker.sh run` bind-mounts
`~/LaneTracking` and `~/wil_project`, so editing on the host and rebuilding
with `ws_build` never means rebuilding the image.

## What is deliberately left out

**Simulation.** Gazebo Fortress, `ros_gz`, `aws-robomaker-small-warehouse-world`
and `gz_kinematic_trajectory` are not installed. ROS 2 Humble has no focal
binaries for `ros_gz`, and the real rig does not simulate.
`vslam_navigation`'s real-rig launches (`mapping_real`, `localize_real`,
`navigate_real`, `nav2_real`) need none of it. `ws_build` drops a
`COLCON_IGNORE` into the warehouse-world package.

**Nav2 pieces `nav2_real.yaml` does not name:** `nav2_amcl` (RTAB-Map
localizes), the DWB controller plugins `dwb_core`, `dwb_critics`,
`dwb_plugins` and `costmap_queue` (regulated pure pursuit is configured),
`nav2_collision_monitor`, `nav2_waypoint_follower`,
`nav2_constrained_smoother`, `nav2_system_tests`, and `nav2_mppi_controller`
(which needs `xtensor`, for which focal has no apt package). Note that
`nav_2d_msgs` and `nav_2d_utils` *are* built even though they sit in the
`nav2_dwb_controller/` directory — `nav2_controller` and `nav2_smoother`
depend on them.

**Do not add `nav2_bringup` to that package set.** It depends on the
`navigation2` metapackage, so asking colcon for it pulls in *every* nav2
package and quietly undoes the whole subset above — MPPI's missing `xtensor`
is what then breaks the build. `nav2_common` is built instead, and that is
what supplies the `RewrittenYaml` which `nav2_real.launch.py` imports.
`vslam_navigation` lists `nav2_bringup` as an `exec_depend`, which does not
affect the build, and none of the real-rig launch files import it.

**RTAB-Map:** the Qt GUI and `rtabmap-databaseViewer` are off (`WITH_QT=OFF`,
`BUILD_TOOLS=OFF`). Nothing in the real-rig path needs them at RUN time:
`pose_gate.py` reads the `.db` with plain `sqlite3`, and the map is watched in
RViz through `/map`, `/cloud_map` and `/mapPath`. `rtabmap_odom`,
`rtabmap_viz` and `rtabmap_rviz_plugins` are not built, per
`rtabmap_ros/MEMORY_CLAUDE/README.md` §4.

It does, however, cost one patch at CONFIGURE time, which is easy to miss:
`rtabmap_util/CMakeLists.txt:31` had `find_package(RTABMap COMPONENTS gui
REQUIRED)` even though the package includes no `rtabmap/gui` header and links
no `rtabmap_gui` target anywhere in `src/` or `include/`. Against a core built
`WITH_QT=OFF` that fails outright:

```
Unsupported or not found required component: gui
```

It never surfaced on the 22.04 host because apt's `ros-humble-rtabmap` is a
full build that ships the gui component. The `gui` requirement is dropped in
the fork (one line, commented in place); the alternative was `WITH_QT=ON`,
pulling Qt5 and PCL-visualization/VTK into the image for something unused.
Worth raising with introlab upstream rather than carrying as a fork delta.

## Decisions worth knowing about

**`libopencv-dev` is never installed.** The base carries a CUDA-enabled OpenCV
4.5.0 as `opencv-dev`/`opencv-libs` in `/lib`, with its CMake config in
`/usr/lib/cmake/opencv4`. apt's `libopencv-dev` is 4.5.4 and would sit beside
it, so RTAB-Map, ORB-SLAM3, VINS and `cv_bridge` could link two different
OpenCVs — the link-time conflict recorded in
`rtabmap_ros/MEMORY_CLAUDE/README.md` §1. For the same reason `ultralytics` is
installed with `--no-deps`: its dependency list would pull `opencv-python` and
a CPU-only torch over the top of both.

**g2o is not optional.** Without g2o or GTSAM, RTAB-Map falls back to TORO,
which cannot carry gravity constraints, and the real-rig map depends on them
(`Mem/UseOdomGravity` + `Optimizer/GravitySigma`; fix 7 in that same file took
the graph from 0 to 164 gravity links). focal has no `libg2o-dev`, so it is
built from the `20201223` release.

**RTAB-Map 0.23.7, not newer.** wil_project's `rtabmap_ros` fork sits on
upstream `2eef2b3`, which requires RTABMap ≥ 0.23.5; 0.23.7 is what it was
validated against. Later fork commits need 0.23.10.

**`/opt/ros/humble/setup.bash` is a shim.** The base built ROS from source, so
the real script is `/opt/ros/humble/install/setup.bash`. 15 files across
wil_project source the conventional path, so the Dockerfile writes a one-line
file there that sources the real one.

**`/home/ambushee` is a symlink to `/home/ubuntu`.** wil_project hardcodes
`/home/ambushee/wil_project` in 13 code and config files — the VINS
`stereo_imu.yaml` configs, two VINS launch files, `dynamic_detector_node.py`
and two scripts. The symlink resolves all of them, so the host checkout needs
no edits and stays diff-clean.

**This image is not portable to a different CPU.** `thirdparty/ORB_SLAM3`
compiles with `-march=native`, and `orbslam3_ros2` matches it deliberately:
with the flag, Eigen's maximum alignment goes from 16 to 64 bytes, which
changes `sizeof(ORB_SLAM3::Frame)` from 149360 to 149376. `SlamWrapper` reads
`tracker->mCurrentFrame` directly, so a mismatch is an ODR violation that
reads a 149 kB object at the wrong offset — undefined behaviour with no
compile error, found originally only through AddressSanitizer
(`orbslam3_ros2/MEMORY_CLAUDE/integration.md` §7.13).

Here the library is compiled during `docker build` and the node during
`ws_build`, both on the same Jetson, so `-march=native` resolves identically
and the pair agrees. But build the image on one machine and run `ws_build` on
another with a different microarchitecture and §7.13 returns silently. If the
image is ever moved, rebuild it — do not just re-run `ws_build`.

**ORB-SLAM3's revision comes from the submodule, not from the script.**
`build_orbslam3.sh` hardcodes `ORB_COMMIT=5c2b00f`, which is an ancestor of
the `thirdparty/ORB_SLAM3` gitlink (`a639fa4` at the time of writing).
`orbslam3_ros2/src/slam_wrapper.cpp` calls `ConfigureEvaluationLogging`,
`GetEvaluationState`, `SaveEvaluationTrajectories`, `SetBoxVoteMinAge`,
`BoxVote` and an extended `TrackStereo`, none of which exist at the script's
pin, so building it yields "class ORB_SLAM3::System has no member named ...".
`docker.sh` therefore reads the gitlink with
`git rev-parse HEAD:thirdparty/ORB_SLAM3` and passes it as the
`ORB_SLAM3_COMMIT` build arg. **This also means a plain `./build_orbslam3.sh`
on the host is currently broken** — fix `ORB_COMMIT` there if you build
outside Docker.

**`ROS_DOMAIN_ID=29`,** LaneTracking's value. wil_project's compose defaulted
to 0. Host and container must agree.

**Two separate workspace overlays**, sourced wil_project first then
LaneTracking:

```
/opt/ros/humble/install  ->  /opt/wil/rosdeps  ->  /opt/wil/nav2
   ->  ~/wil_project/install_docker  ->  ~/LaneTracking/ros2_ws/install_docker
```

**Both** workspaces build into `build_docker/`, `install_docker/` and
`log_docker/`, so a build made on a host (or by an earlier root-owned
container) keeps its own `build/`, `install/` and `log/` untouched. For
LaneTracking there is a second reason: this repository's `ros2_ws/build`,
`install` and `log` are owned by `root:root` from earlier container runs that
executed as root, and a uid-1000 build cannot write into them --
`PermissionError: [Errno 13] Permission denied: 'log/build_...'`. Writing
elsewhere avoids needing sudo. Clean the stale trio whenever convenient:

```bash
sudo rm -rf ~/LaneTracking/ros2_ws/{build,install,log}
```

The entrypoint and `.bashrc` source `install_docker` and fall back to
`install/`, so a pre-existing host build is still picked up.

**`ros2_ws/src/vins_fusion_ros2` is a dangling symlink** into a tree that was
deleted from the LaneTracking root. VINS now comes from wil_project, so
`ws_build lane` removes the broken link rather than letting colcon trip on it.

## ORB-SLAM3 vocabulary

The ORB-SLAM3 launch files resolve their vocabulary to
`~/wil_project/thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt`, but the submodule
ships only `ORBvoc.txt.tar.gz`. Unpack it once on the host:

```bash
cd ~/wil_project/thirdparty/ORB_SLAM3/Vocabulary && tar xzf ORBvoc.txt.tar.gz
```

That is 145 MB and the submodule gitignores `Vocabulary/ORBvoc.txt`
(`.gitignore:67`), so the checkout stays diff-clean. The image also carries an
unpacked copy at `/opt/wil/thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt`, usable
via `vocabulary:=` if you would rather not unpack in the workspace. The
entrypoint warns when neither is in place.

## RealSense

Both repositories talk to the D455 through `pyrealsense2`, not
`realsense2_camera`, so no librealsense ROS driver is in the image. `docker.sh`
mounts the host's source-built `pyrealsense2` and its `librealsense-src`
build output when they exist — both are needed together, because the `.so`
has an absolute RUNPATH into that build tree. Override with `PYRS_HOST=` and
`LIBRS_HOST=`. See LaneTracking's README §1 for why the pip wheel will not do.

**This base image's ROS is an old Humble snapshot, built from source.** That,
not the Ubuntu version, is the root cause of most of the porting work, and
three separate fixes address it:

* `/opt/ros/humble/setup.bash` is a shim sourcing `install/setup.bash`, because
  15 files across wil_project source the conventional path.
* `/opt/ros/humble/{lib,include,share,bin}` are symlinks into `install/`, for
  packages that hard-code the apt layout. `rtabmap_msgs/CMakeLists.txt:15` is
  the case that forced it -- an aarch64-only block (upstream #1285, #1288) doing
  `find_library(rcutils_LIB ... PATHS "/opt/ros/$ENV{ROS_DISTRO}/lib"
  NO_DEFAULT_PATH REQUIRED)`, where `NO_DEFAULT_PATH` means no environment
  variable can rescue it. It never fires on x86_64.
* Every `.h` under the ROS include tree gets a sibling `.hpp` symlink where none
  exists (2138 of them, purely additive). ROS 2 renamed its headers `.h` ->
  `.hpp` and backported that to Humble, so current Humble ships both; this
  2023 snapshot has only `.h`. The rtabmap_ros fork needs eleven of the new
  spellings -- `cv_bridge/cv_bridge.hpp`,
  `tf2_ros/{buffer,transform_listener,transform_broadcaster}.hpp`,
  `tf2/utils.hpp`, two `image_geometry/*` and four `message_filters/*`.

**A real uid-1000 account exists.** `LaneTracking/Docker/Dockerfile` switches to
`USER 1000` without creating the user, which leaves `uid=1000 gid=0(root)` and
no passwd entry. colcon then dies with
`KeyError: 'getpwuid(): uid not found: 1000'`, and because gid is 0 everything
written into the mounted repositories lands as `<user>:root` on the host. This
image creates group and user 1000, so host files come out `nvidia:nvidia`.

## Build notes

* Expect hours. Nav2, RTAB-Map and ORB-SLAM3 are the bulk; each is its own
  cached layer, so a failure is resumed rather than restarted.
* `build_orbslam3.sh` caps itself at 8 parallel jobs because `-j16` with `-O3`
  OOMs a 15 GB machine — which this Jetson is. Do not run a second heavy build
  alongside it.
* Disk: the base layers are shared with `lanetracking/ros:humble`, so the
  increment is the source builds. Check `df -h /` first.
* `nav2` on humble targets gcc 11 and this base has gcc 9.4. In practice all 20
  packages built unmodified; if that changes, a missing `#include` at that
  layer is the likely cause.
* `ws_build` retries the wil_project workspace at `-j4 -> 2 -> 1`. Memory, not
  CPU, is the binding constraint: this Jetson has 15 GB and only zram swap,
  while `rtabmap_sync`'s `CommonDataSubscriberRGBD2..6.cpp` and ORB-SLAM3's
  `Optimizer.cc` can push a single `cc1plus` past 4 GB RSS, so a parallel wave
  gets the compiler OOM-killed. gcc reports only
  `c++: fatal error: Killed signal terminated program cc1plus` with no
  `error:` line, which reads as a mystery failure. make keeps finished objects,
  so each retry resumes. The retry fires ONLY on that signature -- a real
  compile error fails immediately instead of hiding behind three slow retries.
  Override the ladder with `WS_JOBS_LADDER="2 1"`.

## Running the stacks together

Both repositories are in one container on `--network=host` with
`ROS_DOMAIN_ID=29`, so the two sides see each other's topics with no extra
configuration. Open more shells with `./docker.sh exec`.

### LaneTracking — robot I/O, cameras, lane detection

```bash
ros2 launch robot_bringup robot_lanedetect_control.launch.py \
    show_images:=false \
    ori_width:=480 ori_height:=320 \
    engine_path:=model/tusimple_res34_bend_25_v1_480x320.engine \
    config_path:=config/tusimple_res34_bend_25_v1_480x320.py
```

See LaneTracking's own README for the PID controller and the CAN interface.

### wil_project — VSLAM and navigation (real rig)

Three one-command entry points, all from `vslam_navigation`. `map_name` is the
only required argument; `calibration` selects the config directory used by
*both* the front end and the stereo shim (`wil` is the live calibration,
`wil_311026_selfcal` the self-calibrated 31 Oct mount).

**1. Collect a map** — ORB-SLAM3 stereo-inertial front end + RTAB-Map behind
it, RViz as the GUI. `mode:=construct` starts a new database (wiping it),
`mode:=extend` appends a session to one that must already exist.

```bash
ros2 launch vslam_navigation mapping_real.launch.py map_name:=realv7_002
# in another shell, for bag replay -- NO --clock, the bags already record one:
ros2 bag play "<bag>" --rate 0.5
```

Results land in `map/<map_name>/<map_name>.db` and `map/<map_name>/vio.csv`,
with the ORB trajectory under `output/output_orb/real/<map_name>_mapping/`.

**2. Relocalise on a prior map** — RTAB-Map opens the database read-only and
adds no nodes; once a frame is recognised it publishes the `map ->
orbslam3_world` correction and `/localization_pose`.

```bash
ros2 launch vslam_navigation localize_real.launch.py map_name:=realv7_001_prior
ros2 bag play "<another bag of the same site>" --rate 0.5
```

`calibration` **must** match the one the prior map was built with — the stored
visual words come from the rectified images. Until the first recognition the
robot exists only in the odometry frame; that is expected, not a failure. Note
from their design record: a localisation covariance under 9999 is *not* a
correctness signal (one run reported 98 % localised with poses 161 m outside a
14 x 21 m building). Check `/localization_pose` falls inside the map extent and
that `map -> orbslam3_world` stays stable between recognitions.

**3. Navigate** — the lot: ORB-SLAM3 + RTAB-Map localize + Nav2 with the pose
gate and the Ackermann bridge.

```bash
# dry run (bridge_dry_run defaults true -- logs commands instead of sending)
ros2 launch vslam_navigation navigate_real.launch.py map_name:=realv7_001_prior

# on the robot, live: publishes /cmd_velocity and /cmd_steering
ros2 launch vslam_navigation navigate_real.launch.py map_name:=<prior> \
    use_sim_time:=false bridge_dry_run:=false
```

`use_sim_time:=true` (the default) is for bag replay, because the bags record
`/clock`; pass `false` when running live. **Leave `bridge_dry_run` true until
`wheel_base` and the topic units are verified on your rig** — it is what stops
Nav2 from driving the robot while you are still checking the chain. The pose
gate starts Nav2 only once localisation is stable, and zeroes the bridge
whenever it drops. Set a goal with RViz's 2D Goal Pose (`/goal_pose`).

Nav2 can also be exercised with no robot and no estimator, serving the grid
from a map file and standing in static transforms for the SLAM chain:

```bash
ros2 launch vslam_navigation nav2_real.launch.py use_sim_time:=false \
    map_yaml:=/path/prior.yaml database:=map/<m>/<m>.db bridge_dry_run:=true
```

### What will not run here

`wil_cmd` and `wil_cmd_v2` in wil_project are mostly **simulation** recipes —
`aws_robomaker_small_warehouse_world` launches, `script/viewer.py` against a
`.world` file, and the `wil_sim_*` variants. This image has no Gazebo
Fortress and no `ros_gz`, so those do not work in it; see
[What is deliberately left out](#what-is-deliberately-left-out). The real-rig
launches above supersede the older two-terminal procedure of running
`orbslam3_ros2 wil_stereo_imu.launch.py` and `rtabmap.launch.py` separately.

Two operational notes carried over from their records: RTAB-Map is started a
few seconds after ORB-SLAM3 on purpose (stereo pairs arriving before ORB has
initialised its IMU each block the back end for the full 5 s
`wait_for_transform`, and a handful of those yield a run with zero nodes), and
this pipeline is thermally sensitive — at 92 °C a throttled CPU ran the same
pipeline fifteen times slower.
