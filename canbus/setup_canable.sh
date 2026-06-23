#!/bin/bash

# Setup script for CANable/CANable2 USB CAN adapter

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CAN_INTERFACE="can0"
CAN_BITRATE="500000"  # 500 kbps default
SERIAL_DEVICE="/dev/ttyCANable"
CREATE_SERVICE=false
VERBOSE=false

# Function to print colored output
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then 
        print_message $RED "Error: Please run as root (use sudo)"
        exit 1
    fi
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -i, --interface <name>    CAN interface name (default: can0)
    -b, --bitrate <rate>      CAN bitrate in bps (default: 500000)
    -d, --device <path>       Serial device path (default: /dev/ttyCANable)
    -s, --service            Create systemd service for boot persistence
    -v, --verbose            Enable verbose output
    -h, --help               Display this help message

EXAMPLES:
    sudo $0                           # Setup can0 at 500kbps
    sudo $0 -i can1 -b 1000000      # Setup can1 at 1Mbps
    sudo $0 -s                       # Setup with systemd service
    sudo $0 -d /dev/ttyACM0          # Use specific device

SUPPORTED BITRATES:
    10000   - 10 kbps
    20000   - 20 kbps
    50000   - 50 kbps
    100000  - 100 kbps
    125000  - 125 kbps
    250000  - 250 kbps
    500000  - 500 kbps (default)
    750000  - 750 kbps
    1000000 - 1 Mbps

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--interface)
            CAN_INTERFACE="$2"
            shift 2
            ;;
        -b|--bitrate)
            CAN_BITRATE="$2"
            shift 2
            ;;
        -d|--device)
            SERIAL_DEVICE="$2"
            shift 2
            ;;
        -s|--service)
            CREATE_SERVICE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_message $RED "Unknown option: $1"
            usage
            ;;
    esac
done

# Get the actual username (not root when using sudo)
ACTUAL_USER="${SUDO_USER:-$USER}"

# Function to check if CAN is already setup
check_existing_setup() {
    print_message $BLUE "=== Checking Existing Setup ==="
    
    # Check if udev rule exists
    if [ -f "/etc/udev/rules.d/99-canable.rules" ]; then
        print_message $GREEN "✓ udev rules already exist"
    fi
}

# Function to check CANable device
check_canable_device() {
    print_message $BLUE "=== Checking for CANable Device ==="
    
    # Check if CANable is connected via USB
    if lsusb | grep -q "16d0:117e"; then
        print_message $GREEN "✓ CANable2 device found via USB"
        if $VERBOSE; then
            lsusb | grep "16d0:117e"
        fi
        return 0
    elif lsusb | grep -q "16d0:10e8"; then
        print_message $GREEN "✓ CANable device found via USB"
        if $VERBOSE; then
            lsusb | grep "16d0:10e8"
        fi
        return 0
    else
        print_message $YELLOW "⚠ No CANable device found via USB"
        print_message $YELLOW "Available USB devices:"
        lsusb
        return 1
    fi
}

# Function to create udev rules
create_udev_rules() {
    print_message $BLUE "=== Creating udev Rules ==="
    
    local UDEV_RULE_FILE="/etc/udev/rules.d/99-canable.rules"
    
    # Check if rules already exist
    if [ -f "$UDEV_RULE_FILE" ]; then
        print_message $YELLOW "udev rules already exist, updating..."
    fi
    
    cat > "$UDEV_RULE_FILE" << 'EOF'
# CANable/CANable2 USB CAN adapter rules

# CANable2 (newer version)
SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="117e", SYMLINK+="ttyCANable", MODE="0666", GROUP="dialout"

# Original CANable
SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="10e8", SYMLINK+="ttyCANable", MODE="0666", GROUP="dialout"

# Alternative rule using product string
SUBSYSTEM=="tty", ATTRS{product}=="CANable*", SYMLINK+="ttyCANable", MODE="0666", GROUP="dialout"

# Rule for any ttyACM device from these vendors (fallback)
SUBSYSTEM=="tty", KERNEL=="ttyACM[0-9]*", ATTRS{idVendor}=="16d0", MODE="0666", GROUP="dialout"
EOF

    print_message $GREEN "✓ Created udev rules"
    
    # Add user to dialout group
    if ! groups "$ACTUAL_USER" | grep -q "dialout"; then
        print_message $YELLOW "Adding $ACTUAL_USER to dialout group..."
        usermod -a -G dialout "$ACTUAL_USER"
        print_message $GREEN "✓ Added $ACTUAL_USER to dialout group"
        print_message $YELLOW "Note: User needs to log out and back in for group change to take effect"
    else
        print_message $GREEN "✓ User $ACTUAL_USER already in dialout group"
    fi
    
    # Reload udev rules
    print_message $YELLOW "Reloading udev rules..."
    udevadm control --reload-rules
    udevadm trigger
    print_message $GREEN "✓ Reloaded udev rules"
}

# Function to find CANable device
find_canable_device() {
    # First try the symlink
    if [ -L "/dev/ttyCANable" ]; then
        SERIAL_DEVICE="/dev/ttyCANable"
        print_message $GREEN "✓ Found CANable at $SERIAL_DEVICE"
        return 0
    fi
    
    # Otherwise, look for ttyACM devices
    for device in /dev/ttyACM*; do
        if [ -e "$device" ]; then
            # Check if it's a CANable by vendor/product ID
            local vendorid=$(udevadm info -q property -n "$device" | grep ID_VENDOR_ID | cut -d= -f2)
            local productid=$(udevadm info -q property -n "$device" | grep ID_MODEL_ID | cut -d= -f2)
            
            if [[ "$vendorid" == "16d0" ]] && ([[ "$productid" == "117e" ]] || [[ "$productid" == "10e8" ]]); then
                SERIAL_DEVICE="$device"
                print_message $GREEN "✓ Found CANable at $SERIAL_DEVICE"
                return 0
            fi
        fi
    done
    
    print_message $RED "✗ No CANable device found"
    return 1
}

# Function to verify setup
verify_can_setup() {
    print_message $BLUE "=== Verifying CAN Setup ==="
    
    # Show interface details
    print_message $YELLOW "\nInterface details:"
    ls /dev/ttyCAN*
    
    return 0
}

# Function to create systemd service
create_systemd_service() {
    print_message $BLUE "=== Creating Systemd Service ==="
    
    local service_file="/etc/systemd/system/canable-setup.service"
    
    cat > $service_file << EOF
[Unit]
Description=CANable Setup
After=multi-user.target
Wants=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/setup_canable_auto.sh
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # Create auto-setup script
    cat > /usr/local/bin/setup_canable_auto.sh << EOF
#!/bin/bash
# Auto-setup script for CANable

# Wait for device to be available
for i in {1..30}; do
    if [ -e "$SERIAL_DEVICE" ] || [ -e "/dev/ttyCANable" ]; then
        break
    fi
    sleep 1
done

# Find device if using symlink
if [ -L "/dev/ttyCANable" ]; then
    DEVICE="/dev/ttyCANable"
else
    DEVICE="$SERIAL_DEVICE"
fi

# Wait for interface
sleep 1

# Bring up interface
ip link set $CAN_INTERFACE up
ip link set $CAN_INTERFACE txqueuelen 1000

# Keep running
while true; do
    if ! ip link show $CAN_INTERFACE | grep -q "state UP"; then
        echo "Interface down, restarting..."
        exit 1
    fi
    sleep 10
done
EOF

    chmod +x /usr/local/bin/setup_canable_auto.sh
    
    systemctl daemon-reload
    systemctl enable canable-setup.service
    
    print_message $GREEN "✓ Systemd service created and enabled"
}

# Function to display test commands
show_test_commands() {
    print_message $BLUE "\n=== Quick Commands ==="
 
    print_message $YELLOW "Device location:"
    if [ -L "/dev/ttyCANable" ]; then
        echo "   /dev/ttyCANable -> $(readlink /dev/ttyCANable)"
    else
        echo "   $SERIAL_DEVICE"
    fi
}

# Main execution
main() {
    print_message $GREEN "=== CANable USB CAN Setup Script ==="
    print_message $YELLOW "Interface: $CAN_INTERFACE | Bitrate: $CAN_BITRATE bps"
    echo
    
    # Check root privileges
    check_root
    
    # Check if already setup
    check_existing_setup
    
    # Check for CANable device
    if ! check_canable_device; then
        print_message $RED "Please connect your CANable device and try again"
        exit 1
    fi
       
    # Create udev rules
    create_udev_rules
    
    # Wait for udev to process
    sleep 2
    
    # Find CANable device
    if ! find_canable_device; then
        print_message $YELLOW "Please unplug and replug your CANable device, then run this script again"
        exit 1
    fi

    # Verify setup
    if verify_can_setup; then
        print_message $GREEN "\n✓ CANable setup completed successfully!"
        
        # Create systemd service if requested
        if [[ "$CREATE_SERVICE" = true ]]; then
            create_systemd_service
        fi
        
        # Show test commands
        show_test_commands
        
    else
        print_message $RED "\n✗ CANable setup failed!"
        exit 1
    fi
}

# Run main function
main