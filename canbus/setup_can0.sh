#!/bin/bash

# CAN Bus Setup Script for Hardware CAN0 Interface

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
ENABLE_LOOPBACK=false
VERBOSE=false
CREATE_SERVICE=false

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
    -l, --loopback           Enable loopback mode for testing
    -s, --service            Create systemd service for boot persistence
    -v, --verbose            Enable verbose output
    -h, --help               Display this help message

EXAMPLES:
    sudo $0                           # Setup can0 at 500kbps
    sudo $0 -i can1 -b 1000000      # Setup can1 at 1Mbps
    sudo $0 -l                       # Setup can0 in loopback mode
    sudo $0 -s                       # Setup with systemd service
    sudo $0 -i can0 -b 250000 -v    # Setup can0 at 250kbps with verbose

SUPPORTED BITRATES:
    125000  - 125 kbps
    250000  - 250 kbps
    500000  - 500 kbps (default)
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
        -l|--loopback)
            ENABLE_LOOPBACK=true
            shift
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

# Function to check if CAN is already setup
check_existing_setup() {
    print_message $BLUE "=== Checking Existing Setup ==="
    
    # Check if interface exists and is up
    if ip link show $CAN_INTERFACE &> /dev/null; then
        if ip link show $CAN_INTERFACE | grep -q "state UP"; then
            # Get current bitrate if physical CAN
            if [[ ! "$ENABLE_LOOPBACK" = true ]]; then
                local current_bitrate=$(ip -details link show $CAN_INTERFACE | grep -oP 'bitrate \K[0-9]+' || echo "unknown")
                if [ "$current_bitrate" = "$CAN_BITRATE" ]; then
                    print_message $GREEN "✓ CAN interface $CAN_INTERFACE is already configured with bitrate $CAN_BITRATE"
                    print_message $YELLOW "Skipping setup. Use 'ip link set $CAN_INTERFACE down' to reconfigure."
                    
                    # Still show test commands
                    show_test_commands
                    exit 0
                else
                    print_message $YELLOW "Interface $CAN_INTERFACE exists with different bitrate ($current_bitrate)"
                    print_message $YELLOW "Reconfiguring to $CAN_BITRATE bps..."
                fi
            else
                print_message $GREEN "✓ Virtual CAN interface $CAN_INTERFACE is already up"
                print_message $YELLOW "Skipping setup."
                show_test_commands
                exit 0
            fi
        fi
    fi
}

# Function to check system
check_system() {
    print_message $BLUE "=== System Check ==="
    
    # Check if running on Jetson
    if [ -f /etc/nv_tegra_release ]; then
        print_message $GREEN "✓ Running on NVIDIA Jetson platform"
        if $VERBOSE; then
            cat /etc/nv_tegra_release
        fi
    else
        print_message $YELLOW "⚠ Not running on Jetson platform, continuing anyway..."
    fi
    
    # Check kernel CAN support
    if [ -d /sys/class/net ] && grep -q "^can$" /proc/modules 2>/dev/null; then
        print_message $GREEN "✓ CAN support detected in kernel"
    else
        print_message $YELLOW "⚠ CAN modules not loaded yet"
    fi
}

# Function to install required packages
install_dependencies() {
    print_message $BLUE "=== Installing Dependencies ==="
    
    local packages=("can-utils" "iproute2" "kmod" "python3-pip")
    local need_install=false
    
    for pkg in "${packages[@]}"; do
        if ! dpkg -l | grep -q "^ii  $pkg"; then
            need_install=true
            break
        fi
    done
    
    if $need_install; then
        print_message $YELLOW "Installing required packages..."
        apt-get update
        apt-get install -y "${packages[@]}"
        print_message $GREEN "✓ Dependencies installed successfully"
    else
        print_message $GREEN "✓ All dependencies are already installed"
    fi
}

# Function to load CAN kernel modules
load_can_modules() {
    print_message $BLUE "=== Loading CAN Kernel Modules ==="
    
    local modules=("can" "can-dev" "can-raw" "can-bcm" "can-gw")
    
    for module in "${modules[@]}"; do
        if ! lsmod | grep -q "^$module"; then
            if $VERBOSE; then
                print_message $YELLOW "Loading module: $module"
            fi
            modprobe $module 2>/dev/null || true
        fi
    done
    
    # Load virtual CAN for loopback
    if [[ "$ENABLE_LOOPBACK" = true ]]; then
        modprobe vcan 2>/dev/null || true
    else
        # Load Jetson-specific CAN modules
        modprobe mttcan 2>/dev/null || true
    fi
    
    print_message $GREEN "✓ CAN modules loaded"
    
    if $VERBOSE; then
        print_message $YELLOW "Loaded modules:"
        lsmod | grep -E "^(can|vcan|mttcan)"
    fi
}

# Function to configure CAN interface
configure_can_interface() {
    print_message $BLUE "=== Configuring CAN Interface: $CAN_INTERFACE ==="
    
    # Check if interface exists
    if ip link show $CAN_INTERFACE &> /dev/null; then
        print_message $YELLOW "Interface $CAN_INTERFACE exists, bringing it down..."
        ip link set $CAN_INTERFACE down
    else
        if [[ "$ENABLE_LOOPBACK" = true ]]; then
            print_message $YELLOW "Creating virtual CAN interface: $CAN_INTERFACE"
            ip link add dev $CAN_INTERFACE type vcan
        else
            print_message $RED "Error: Physical CAN interface $CAN_INTERFACE not found"
            print_message $YELLOW "Available network interfaces:"
            ip link show | grep -E "^[0-9]+: " | awk '{print $2}' | tr -d ':'
            print_message $YELLOW "\nTips:"
            print_message $YELLOW "- Check if CAN transceiver is connected"
            print_message $YELLOW "- Verify device tree configuration for CAN"
            print_message $YELLOW "- Try running with -l flag for loopback testing"
            exit 1
        fi
    fi
    
    # Configure interface based on type
    if [[ ! "$ENABLE_LOOPBACK" = true ]]; then
        # Physical CAN configuration
        print_message $YELLOW "Setting bitrate to $CAN_BITRATE bps..."
        
        # Set bitrate
        if ! ip link set $CAN_INTERFACE type can bitrate $CAN_BITRATE; then
            print_message $RED "Failed to set bitrate. Trying alternative method..."
            # Alternative method for some systems
            ip link set $CAN_INTERFACE type can bitrate $CAN_BITRATE sample-point 0.875
        fi
        
        # Set additional parameters for better performance
        ip link set $CAN_INTERFACE type can restart-ms 100 2>/dev/null || true
        
        # Enable CAN FD if supported (optional)
        if $VERBOSE; then
            ip link set $CAN_INTERFACE type can fd on 2>/dev/null && \
                print_message $GREEN "✓ CAN FD enabled" || \
                print_message $YELLOW "⚠ CAN FD not supported"
        fi
        
        # Set txqueuelen for better performance
        ip link set $CAN_INTERFACE txqueuelen 1000
    fi
    
    # Bring interface up
    print_message $YELLOW "Bringing up interface $CAN_INTERFACE..."
    if ! ip link set $CAN_INTERFACE up; then
        print_message $RED "Failed to bring up interface"
        exit 1
    fi
    
    print_message $GREEN "✓ CAN interface $CAN_INTERFACE configured successfully"
}

# Function to verify CAN setup
# Function to verify CAN setup (FIXED VERSION)
verify_can_setup() {
    print_message $BLUE "=== Verifying CAN Setup ==="
    
    # Check if interface is up
    if ip link show $CAN_INTERFACE | grep -q "state UP"; then
        print_message $GREEN "✓ Interface $CAN_INTERFACE is UP"
    else
        print_message $RED "✗ Interface $CAN_INTERFACE is DOWN"
        return 1
    fi
    
    # Show interface details
    print_message $YELLOW "\nInterface details:"
    ip -details link show $CAN_INTERFACE
    
    # Show CAN statistics
    if [[ ! "$ENABLE_LOOPBACK" = true ]]; then
        print_message $YELLOW "\nCAN statistics:"
        ip -statistics link show $CAN_INTERFACE
    fi
    
    # Check for CAN errors - FIXED GREP PATTERN
    local can_state=$(ip -details link show $CAN_INTERFACE | grep -oP 'can state \K[A-Z-]+' || echo "UNKNOWN")
    
    case "$can_state" in
        "ERROR-ACTIVE")
            print_message $GREEN "✓ CAN bus state: $can_state (Normal operation)"
            ;;
        "ERROR-WARNING")
            print_message $YELLOW "⚠ CAN bus state: $can_state (Elevated error count)"
            ;;
        "ERROR-PASSIVE")
            print_message $YELLOW "⚠ CAN bus state: $can_state (High error count - check connections)"
            ;;
        "BUS-OFF")
            print_message $RED "✗ CAN bus state: $can_state (Bus disabled due to errors)"
            ;;
        *)
            print_message $YELLOW "⚠ CAN bus state: $can_state"
            ;;
    esac
    
    # Show error counters
    local error_info=$(ip -details link show $CAN_INTERFACE | grep -oP 'berr-counter tx \K[0-9]+ rx [0-9]+' || echo "")
    if [[ -n "$error_info" ]]; then
        print_message $YELLOW "Error counters: TX/RX = $error_info"
    fi
    
    return 0
}

# Function to create systemd service for persistent setup
create_systemd_service() {
    print_message $BLUE "=== Creating Systemd Service ==="
    
    local service_file="/etc/systemd/system/can-setup.service"
    
    cat > $service_file << EOF
[Unit]
Description=CAN Bus Setup for Autonomous Vehicle
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/setup_can_auto.sh
RemainAfterExit=yes
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # Create auto-setup script
    cat > /usr/local/bin/setup_can_auto.sh << EOF
#!/bin/bash
# Auto-setup script for CAN interface

# Load modules
modprobe can
modprobe can-dev
modprobe can-raw
modprobe mttcan 2>/dev/null || true

# Wait for interface to be available
for i in {1..10}; do
    if ip link show $CAN_INTERFACE &> /dev/null; then
        break
    fi
    sleep 1
done

# Setup interface
ip link set $CAN_INTERFACE down 2>/dev/null || true
ip link set $CAN_INTERFACE type can bitrate $CAN_BITRATE
ip link set $CAN_INTERFACE type can restart-ms 100
ip link set $CAN_INTERFACE txqueuelen 1000
ip link set $CAN_INTERFACE up

# Log status
if ip link show $CAN_INTERFACE | grep -q "state UP"; then
    echo "CAN interface $CAN_INTERFACE started successfully"
    exit 0
else
    echo "Failed to start CAN interface $CAN_INTERFACE"
    exit 1
fi
EOF

    chmod +x /usr/local/bin/setup_can_auto.sh
    
    systemctl daemon-reload
    systemctl enable can-setup.service
    
    print_message $GREEN "✓ Systemd service created and enabled"
    print_message $YELLOW "Service will start on next boot"
}

# Function to display test commands
show_test_commands() {
    print_message $BLUE "\n=== Quick Test Commands ==="
    
    print_message $YELLOW "1. Monitor CAN traffic:"
    echo "   candump $CAN_INTERFACE"
    echo ""
    
    print_message $YELLOW "2. Monitor with timestamps:"
    echo "   candump -ta $CAN_INTERFACE"
    echo ""
    
    print_message $YELLOW "3. Send test message (vehicle stop):"
    echo "   cansend $CAN_INTERFACE 202#0000000000000000"
    echo ""
    
    print_message $YELLOW "4. Send forward 2m/s, straight:"
    echo "   cansend $CAN_INTERFACE 202#0000004000000000"
    echo ""
    
    print_message $YELLOW "5. Monitor specific CAN IDs (control & feedback):"
    echo "   candump $CAN_INTERFACE,202:7FF,182:7FF"
    echo ""
    
    print_message $YELLOW "6. Generate test traffic (10Hz):"
    echo "   cangen $CAN_INTERFACE -I 202 -D 0000004000000000 -g 100 -L 8"
    echo ""
    
    if [[ "$ENABLE_LOOPBACK" = true ]]; then
        print_message $BLUE "\n=== Loopback Test ==="
        echo "Terminal 1: candump $CAN_INTERFACE"
        echo "Terminal 2: cansend $CAN_INTERFACE 123#DEADBEEF"
    fi
}

# Function to perform basic CAN test
perform_can_test() {
    print_message $BLUE "\n=== Performing Basic CAN Test ==="
    
    # For loopback, we can test send/receive
    if [[ "$ENABLE_LOOPBACK" = true ]]; then
        # Start candump in background
        timeout 2 candump -n 1 $CAN_INTERFACE > /tmp/can_test.log 2>&1 &
        local DUMP_PID=$!
        
        sleep 0.5
        
        # Send test message
        if cansend $CAN_INTERFACE 7FF#DEADBEEF 2>/dev/null; then
            print_message $GREEN "✓ Successfully sent test message"
            
            # Check if received
            sleep 0.5
            if grep -q "7FF" /tmp/can_test.log 2>/dev/null; then
                print_message $GREEN "✓ Successfully received test message"
            fi
        else
            print_message $RED "✗ Failed to send test message"
            return 1
        fi
        
        # Cleanup
        kill $DUMP_PID 2>/dev/null || true
        rm -f /tmp/can_test.log
    else
        # For physical CAN, just test send
        if cansend $CAN_INTERFACE 7FF#DEADBEEF 2>/dev/null; then
            print_message $GREEN "✓ Successfully sent test message"
            print_message $YELLOW "Note: Physical CAN requires another node to verify reception"
        else
            print_message $RED "✗ Failed to send test message"
            print_message $YELLOW "This might be normal if no other node is connected"
        fi
    fi
    
    return 0
}

# Main execution
main() {
    print_message $GREEN "=== CAN Bus Setup Script for Autonomous Vehicle ==="
    print_message $YELLOW "Interface: $CAN_INTERFACE | Bitrate: $CAN_BITRATE bps"
    if [[ "$ENABLE_LOOPBACK" = true ]]; then
        print_message $YELLOW "Mode: Loopback (Virtual CAN)"
    else
        print_message $YELLOW "Mode: Physical CAN"
    fi
    echo
    
    # Check root privileges
    check_root
    
    # Check if already setup
    check_existing_setup
    
    # Check system
    check_system
    
    # Install dependencies
    install_dependencies
    
    # Load kernel modules
    load_can_modules
    
    # Configure CAN interface
    configure_can_interface
    
    # Verify setup
    if verify_can_setup; then
        print_message $GREEN "\n✓ CAN setup completed successfully!"
        
        # Perform basic test
        perform_can_test
        
        # Create systemd service if requested
        if [[ "$CREATE_SERVICE" = true ]] && [[ ! "$ENABLE_LOOPBACK" = true ]]; then
            create_systemd_service
        fi
        
        # Show test commands
        show_test_commands
        
    else
        print_message $RED "\n✗ CAN setup failed!"
        exit 1
    fi
}

# Run main function
main