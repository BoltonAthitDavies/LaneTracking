#!/bin/bash

# Script to install Python dependencies for CANable/CAN communication
# Compatible with Python 3.8+

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Installing Python CAN Dependencies ===${NC}"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python3 --version

# Update pip
echo -e "${YELLOW}Updating pip...${NC}"
python3 -m pip install --upgrade pip

# Install Python CAN packages
echo -e "${YELLOW}Installing python-can and related packages...${NC}"
pip3 install --upgrade \
    python-can \
    "python-can[serial]" \
    pyserial \
    cantools \
    can-isotp

# Install optional but useful packages
echo -e "${YELLOW}Installing optional packages...${NC}"
pip3 install --upgrade \
    numpy \
    matplotlib \
    pandas

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
python3 -c "import can; print(f'python-can version: {can.__version__}')"
python3 -c "import serial; print(f'pyserial version: {serial.__version__}')"

echo -e "${GREEN}✓ All Python dependencies installed successfully!${NC}"

# Show available interfaces
echo -e "\n${YELLOW}Available CAN interfaces:${NC}"
python3 -c "import can; print('\n'.join(can.interfaces.VALID_INTERFACES))"

echo -e "\n${YELLOW}To test the installation, run:${NC}"
echo "python3 test_canable.py"