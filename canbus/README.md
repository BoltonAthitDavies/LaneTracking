# CAN Communication Guide

This guide provides comprehensive CAN bus communication specifications for controlling an autonomous vehicle using NXP microcontroller with Flipsky BLDC Belt Motor 6384 190KV and RX-64 servo motor. It includes automated setup scripts, protocol specifications, and hardware configuration instructions.

## Table of Contents
- [Features](#features)
- [Dependencies](#dependencies)
- [Hardware Components](#hardware-components)
- [Hardware Setup and Configuration](#hardware-setup-and-configuration)
- [CAN Message Protocol](#can-message-protocol)
- [Installation](#installation)
- [Usage](#usage)
- [Unsetup](#unsetup)
- [Troubleshooting](#troubleshooting)
- [Feedback](#feedback)

## Features
- Direct CAN communication with VESC motor controllers
- Real-time vehicle control (velocity and steering)
- Configurable CAN bitrate up to 1 Mbps
- Automated setup script with systemd service
- Support for dual motor control (left and right)
- Feedback reception for actual velocity and steering angle
- Hardware-level safety features and emergency stop support
- Support for both hardware CAN interface and CANable USB adapter

## Dependencies
- Linux kernel with SocketCAN support
- Python 3.8+ (for control scripts)
- can-utils
- iproute2
- kmod (kernel module tools)
- python-can (for CANable support)
- pyserial (for CANable support)

## Hardware Components

### 1. Microcontroller Unit (MCU)
- **Model**: NXP Board
- **CAN Channels**: 2 channels
- **Purpose**: Central control unit for vehicle communication

### 2. Motors
- **Model**: Flipsky BLDC Belt Motor 6384 190KV
- **Power**: 4000W
- **Pulley**: 10mm shaft
- **Quantity**: 2 (left and right)

### 3. Servo Motor (Steering)
- **Model**: RX-64
- **Operating Range**: 0° - 300° (or endless turn mode)
- **Operating Voltage**: 12V - 18.5V
- **Communication**: RS485 Asynchronous Serial
- **ID Range**: 0-253 (default: 254)

### 4. CAN Transceiver Options

#### Option A: Hardware CAN with TJA1050
- **Model**: TJA1050 CAN Bus Interface Module (SMD3)
- **Operating Voltage**: 5V
- **Data Rate**: Up to 1 Mbps
- **Features**: Built-in 120Ω termination resistor (check jumper)

#### Option B: CANable USB Adapter
- **Model**: CANable/CANable 2.0
- **Interface**: USB to CAN protocol
- **Data Rate**: Up to 1 Mbps
- **Features**: Plug-and-play, no wiring required

## Hardware Setup and Configuration

### Option 1: CAN Bus Wiring with TJA1050

| TJA1050 Pin | Connection | Notes |
|-------------|------------|-------|
| VCC | 3.3V | From Jetson or external supply |
| GND | Ground | Common ground with all devices |
| CAN_H | CAN High Bus | Connect to all CAN_H |
| CAN_L | CAN Low Bus | Connect to all CAN_L |
| TXD |  Jetson CAN_TX |  |
| RXD |  Jetson CAN_RX |  |

### Jetson Orin Nano Connections
You can check Jetson Datasheet [here](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_specification_sp.pdf).
| Jetson Pin | Connection |
|------------|------------|
| J17 Pin 1 (CAN_RX) | RXD |
| J17 Pin 2 (CAN_TX) | TXD |
| 3.3V | VCC |
| GND | GND |

### Option 2: CANable USB Connection
Simply plug the CANable adapter into any available USB port. No additional wiring required.

## CAN Message Protocol

### CAN ID 0x202 - Vehicle Control (Transmit)
Sends velocity and steering commands to the vehicle.

**Message Format (8 bytes)**
```
[Velocity (4 bytes)] [Steering (4 bytes)]
```

**Data Specifications**

| Parameter | Min Value | Max Value | Data Type | Description |
|-----------|-----------|-----------|-----------|-------------|
| Velocity | -3.0 m/s | +3.0 m/s | Float32, little-endian | Positive = Forward<br>Negative = Reverse |
| Steering | -40.0° | +40.0° | Float32, little-endian | Negative = Left turn<br>Positive = Right turn<br>0.0° = Straight |

**Example CAN Messages**

| Command | CAN Data (Hex) | Description |
|---------|----------------|-------------|
| Forward 3.0 m/s, straight | 00 00 40 40 00 00 00 00 | Maximum forward speed |
| Forward 2.0 m/s, left 20° | 00 00 00 40 00 00 A0 C1 | Medium speed with left turn |
| Forward 1.5 m/s, right 20° | 00 00 C0 3F 00 00 A0 41 | Slow speed with right turn |
| Reverse 3.0 m/s, straight | 00 00 40 C0 00 00 00 00 | Maximum reverse speed |
| Stop (0 m/s), center | 00 00 00 00 00 00 00 00 | Complete stop |

### CAN ID 0x182 - Vehicle Feedback (Receive)
Receives actual velocity and steering angle from the vehicle.

**Message Format (8 bytes)**
```
[Actual Velocity (4 bytes)] [Actual Steering (4 bytes)]
```

### Control Limits Summary

| Parameter | Range | Center/Neutral | Units |
|-----------|-------|----------------|-------|
| Velocity | -3.0 to +3.0 | 0.0 | m/s |
| Steering | -40.0 to +40.0 | 0.0 | degrees |

## Installation

### Installing System Dependencies

#### For Hardware CAN (can0):
```bash
chmod +x setup_can0.sh
sudo ./setup_can0.sh
```

The setup script will automatically:
- Install required packages (can-utils, iproute2, kmod)
- Load necessary kernel modules
- Configure the CAN interface
- Optionally create a systemd service for boot persistence

#### For CANable USB Adapter:
```bash
chmod +x setup_canable.sh
sudo ./setup_canable.sh
```

The CANable setup script will:
- Install required packages including python-can
- Create udev rules for device recognition
- Configure interface
- Set up /dev/ttyCANable symlink
- Optionally create systemd service

### Installing Python Dependencies
```bash
chmod +x install_python_deps.sh
./install_python_deps.sh
```
## Usage

### Basic Usage

#### Hardware CAN Setup:
```bash
# Using the setup script (recommended)
sudo ./setup_can0.sh

# With options
sudo ./setup_can0.sh -i can0 -b 500000 -s  # With systemd service
sudo ./setup_can0.sh -l                     # Loopback mode for testing
```

#### CANable Setup:
```bash
# Using the setup script (recommended)
sudo ./setup_canable.sh

# With options
sudo ./setup_canable.sh -d /dev/ttyACM0       # Specific device
```

### Parameters
The setup scripts support the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| -i, --interface | String | can0 | CAN interface name (can0 only) |
| -b, --bitrate | Integer | 500000 | CAN bitrate in bps |
| -s, --service | Boolean | False | Create systemd service |
| -v, --verbose | Boolean | False | Enable verbose output |
| -l, --loopback | Boolean | False | Enable loopback mode (can0 only) |
| -d, --device | String | /dev/ttyCANable | Serial device (CANable only) |

### Quick Test Commands

**Monitor CAN Traffic**
```bash
# Monitor all messages
candump can0

# Monitor specific CAN IDs
candump can0,202:7FF,182:7FF
```

**Send Control Commands**
```bash
# Stop command
cansend can0 202#0000000000000000

# Forward at 2m/s, straight
cansend can0 202#0000004000000000

# Forward at 1m/s, turn left 20°
cansend can0 202#0000803F000020C1

# Generate periodic messages (100Hz)
cangen can0 -I 202 -D 0000004000000000 -g 10 -L 8
```

**Python Test Script**
```bash
# Send only at 100Hz
python3 test_canable.py send -d 10

# Receive only
python3 test_canable.py receive -d 10

# Both send and receive
python3 test_canable.py both -d 10

# Custom velocity/steering
python3 test_canable.py send -v 1.5 -s 20.0 -d 5
```

## Unsetup

### Removing CAN Configurations

To remove all CAN configurations created by the setup scripts, use the provided unsetup script:

```bash
chmod +x can_cleanup.sh
sudo ./can_cleanup.sh
```

### Unsetup Options

| Option | Description |
|--------|-------------|
| -v, --verbose | Enable verbose output |
| -f, --force | Force removal without prompts |
| -h, --help | Display help message |

### What Gets Removed

**Default Actions (Safe):**
- Stops and disables systemd services (can-setup.service, canable-setup.service)
- Brings down all CAN interfaces
- Removes CANable udev rules
- Unloads CAN kernel modules

### Usage Examples

```bash
# Basic unsetup (recommended)
sudo ./can_cleanup.sh

# Force unsetup without confirmation prompts
sudo ./can_cleanup.sh -f

# Verbose mode to see detailed actions
sudo ./can_cleanup.sh -v
```

### Manual Cleanup Commands

If needed, you can manually run these commands:

```bash
# Remove specific CAN interface
sudo ip link delete can0

# Remove specific kernel module
sudo rmmod can

# Check for remaining CAN services
sudo systemctl status can*

# Check loaded CAN modules
lsmod | grep can
```

### Safety Notes

- The script will ask for confirmation before destructive actions (unless using -f flag)
- Use -p flag only if you don't need can-utils for other projects
- Use -g flag only if you don't need dialout group membership for other devices
- The script provides a cleanup summary showing what's still configured

## Troubleshooting

### CAN Interface Not Found
```bash
# Check if CAN modules are loaded
lsmod | grep can

# Check available network interfaces
ip link show

# Check dmesg for CAN-related messages
dmesg | grep -i can
```

### No CAN Traffic
1. Check termination resistors (120Ω at both ends)
2. Verify wiring connections (CAN_H, CAN_L)
3. Confirm bitrate matches on all nodes
4. Test with loopback mode first:
   ```bash
   sudo ./setup_can0.sh -l
   ```

### CANable Specific Issues
```bash
# Check if device is connected
lsusb | grep 16d0

# Check device permissions
ls -la /dev/ttyCANable
ls -la /dev/ttyACM*

# Check udev rules
cat /etc/udev/rules.d/99-canable.rules
```

### Permission Denied
```bash
# For hardware CAN
sudo ./setup_can0.sh

# For CANable - add user to dialout group
sudo usermod -a -G dialout $USER
# Then logout and login again
```

### CAN Bus Errors
```bash
# Check CAN statistics
ip -details -statistics link show can0

# Check for bus-off state
ip link show can0

# Reset interface
sudo ip link set can0 down
sudo ip link set can0 up
```
## Feedback

If you have any feedback or questions, please create an issue and we will address them there.