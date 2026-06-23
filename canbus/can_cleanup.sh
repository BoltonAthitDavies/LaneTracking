#!/bin/bash

# CAN Unsetup Script - Removes CAN configurations from both setup scripts

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
VERBOSE=false
FORCE=false

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
    -p, --packages           Remove installed packages (can-utils, etc.)
    -g, --group             Remove user from dialout group
    -v, --verbose           Enable verbose output
    -f, --force             Force removal without prompts
    -h, --help              Display this help message

EXAMPLES:
    sudo $0                  # Basic unsetup
    sudo $0 -p -g           # Full unsetup including packages and group
    sudo $0 -f              # Force unsetup without prompts

WARNING:
    This script will remove CAN configurations created by:
    1. Hardware CAN setup script
    2. CANable USB setup script

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -f|--force)
            FORCE=true
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

# Get the actual username
ACTUAL_USER="${SUDO_USER:-$USER}"

# Function to confirm action
confirm_action() {
    if [[ "$FORCE" = true ]]; then
        return 0
    fi
    
    local message=$1
    read -p "$message (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# Function to stop and disable systemd services
remove_systemd_services() {
    print_message $BLUE "=== Removing Systemd Services ==="
    
    # Hardware CAN service
    if systemctl list-unit-files | grep -q "can-setup.service"; then
        print_message $YELLOW "Stopping and disabling can-setup.service..."
        systemctl stop can-setup.service 2>/dev/null || true
        systemctl disable can-setup.service 2>/dev/null || true
        rm -f /etc/systemd/system/can-setup.service
        print_message $GREEN "✓ Removed can-setup.service"
    fi
    
    # CANable service
    if systemctl list-unit-files | grep -q "canable-setup.service"; then
        print_message $YELLOW "Stopping and disabling canable-setup.service..."
        systemctl stop canable-setup.service 2>/dev/null || true
        systemctl disable canable-setup.service 2>/dev/null || true
        rm -f /etc/systemd/system/canable-setup.service
        print_message $GREEN "✓ Removed canable-setup.service"
    fi
    
    # Reload systemd
    systemctl daemon-reload
}

# Function to bring down CAN interfaces
bring_down_interfaces() {
    print_message $BLUE "=== Bringing Down CAN Interfaces ==="
    
    # Find all CAN interfaces
    local can_interfaces=$(ip link show | grep -E "^[0-9]+: (can|vcan)" | awk -F: '{print $2}' | tr -d ' ')
    
    if [ -z "$can_interfaces" ]; then
        print_message $YELLOW "No CAN interfaces found"
    else
        for interface in $can_interfaces; do
            if ip link show $interface | grep -q "state UP"; then
                print_message $YELLOW "Bringing down $interface..."
                ip link set $interface down
                print_message $GREEN "✓ Interface $interface is down"
                
                # Remove virtual CAN interfaces
                if [[ $interface == vcan* ]]; then
                    if confirm_action "Remove virtual interface $interface?"; then
                        ip link delete $interface
                        print_message $GREEN "✓ Removed virtual interface $interface"
                    fi
                fi
            else
                print_message $YELLOW "Interface $interface is already down"
            fi
        done
    fi
}

# Function to remove udev rules
remove_udev_rules() {
    print_message $BLUE "=== Removing udev Rules ==="
    
    if [ -f "/etc/udev/rules.d/99-canable.rules" ]; then
        rm -f /etc/udev/rules.d/99-canable.rules
        print_message $GREEN "✓ Removed /etc/udev/rules.d/99-canable.rules"
        
        # Reload udev rules
        udevadm control --reload-rules
        udevadm trigger
        print_message $GREEN "✓ Reloaded udev rules"
    else
        print_message $YELLOW "No CANable udev rules found"
    fi
}

# Function to unload kernel modules
unload_kernel_modules() {
    print_message $BLUE "=== Unloading CAN Kernel Modules ==="
    
    # List of modules to unload (in reverse order of loading)
    local modules=("vcan" "mttcan" "can-gw" "can-bcm" "can-raw" "can-dev" "can")
    
    for module in "${modules[@]}"; do
        if lsmod | grep -q "^$module"; then
            if $VERBOSE; then
                print_message $YELLOW "Unloading module: $module"
            fi
            rmmod $module 2>/dev/null || true
        fi
    done
    
    print_message $GREEN "✓ CAN modules unloaded"
}

# Function to show cleanup summary
show_cleanup_summary() {
    print_message $BLUE "\n=== Cleanup Summary ==="
    
    # Check what's still configured
    local issues=0
    
    # Check for running CAN interfaces
    if ip link show | grep -q "can[0-9]\|vcan"; then
        print_message $YELLOW "⚠ CAN interfaces still exist (run 'ip link show' to check)"
        ((issues++))
    fi
    
    # Check for loaded modules
    if lsmod | grep -q "^can"; then
        print_message $YELLOW "⚠ CAN modules still loaded (run 'lsmod | grep can' to check)"
        ((issues++))
    fi
    
    # Check for services
    if systemctl list-unit-files | grep -qE "(can-setup|canable-setup)\.service"; then
        print_message $YELLOW "⚠ Systemd services still exist"
        ((issues++))
    fi
    
    if [ $issues -eq 0 ]; then
        print_message $GREEN "✓ All CAN configurations have been removed"
    else
        print_message $YELLOW "Some configurations may still be present"
    fi
    
    # Additional cleanup commands
    print_message $BLUE "\n=== Additional Cleanup Commands ==="
    print_message $YELLOW "If needed, you can manually run:"
    echo "  sudo ip link delete can0     # Remove specific interface"
    echo "  sudo rmmod can               # Remove specific module"
    echo "  sudo systemctl status can*   # Check for CAN services"
}

# Main execution
main() {
    print_message $GREEN "=== CAN Unsetup Script ==="
    
    if [[ "$FORCE" != true ]]; then
        print_message $YELLOW "This will remove CAN configurations. Use -f to skip confirmations."
        if ! confirm_action "Continue with unsetup?"; then
            print_message $YELLOW "Unsetup cancelled"
            exit 0
        fi
    fi
    
    echo
    
    # Check root privileges
    check_root
    
    # Stop and disable systemd services
    remove_systemd_services
    
    # Bring down CAN interfaces
    bring_down_interfaces
    
    # Remove udev rules
    remove_udev_rules
    
    # Unload kernel modules
    unload_kernel_modules
    
    # Show summary
    show_cleanup_summary
    
    print_message $GREEN "\n✓ CAN unsetup completed!"
}

# Run main function
main