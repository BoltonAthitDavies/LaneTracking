#!/usr/bin/env python3
"""
CAN Test Script with Phase Options
- Send only
- Receive only  
- Send and Receive
Fixed to send at 100Hz
"""
import can
import time
import struct
import sys
import argparse
import os
import threading

def connect_can(device='/dev/ttyACM0'):
    """Connect to CAN bus"""
    # Try to find device
    if not os.path.exists(device):
        # Try ttyCANable
        if os.path.exists('/dev/ttyCANable'):
            device = '/dev/ttyCANable'
        else:
            # Try other ttyACM devices
            for i in range(5):
                alt = f'/dev/ttyACM{i}'
                if os.path.exists(alt):
                    device = alt
                    break
    
    print(f"Connecting to {device}...")
    
    try:
        # Try slcan interface first
        bus = can.Bus(interface='slcan', channel=device, bitrate=500000)
        print(f"Connected via slcan on {device}")
        return bus
    except Exception as e:
        print(f"SLCAN failed: {e}")
        # Try serial interface
        try:
            bus = can.Bus(interface='serial', channel=device, bitrate=500000)
            print(f"Connected via serial on {device}")
            return bus
        except Exception as e:
            print(f"Serial failed: {e}")
            raise

def create_command(velocity, steering):
    """Create vehicle command message"""
    # Pack as little-endian floats
    data = struct.pack('<ff', velocity, steering)
    return can.Message(arbitration_id=0x202, data=data, is_extended_id=False)

def send_phase(bus, duration=10, velocity=2.0, steering=40.0):
    """Send phase - send commands at 100Hz"""
    print("\n=== SEND PHASE ===")
    print(f"Duration: {duration} seconds")
    print(f"Send rate: 100Hz")
    print("Commands: 2 m/s, 40° → 0 m/s, 0.01°\n")
    
    # Create messages
    move_msg = create_command(velocity, steering)
    stop_msg = create_command(0.0, 0.01)
    
    start_time = time.time()
    send_count = 0
    last_print_time = 0
    
    while (time.time() - start_time) < duration:
        elapsed = time.time() - start_time
        
        # Alternate between move and stop every 2 seconds
        if int(elapsed) % 4 < 2:
            # Send move command
            bus.send(move_msg)
            current_msg = move_msg
            cmd_str = "2.0 m/s, 40.0°"
        else:
            # Send stop command
            bus.send(stop_msg)
            current_msg = stop_msg
            cmd_str = "0.0 m/s, 0.01°"
        
        send_count += 1
        
        # Print status every second
        if int(elapsed) > last_print_time:
            last_print_time = int(elapsed)
            rate = send_count / max(0.1, elapsed)
            print(f"[{elapsed:5.1f}s] Sending: {cmd_str} | Rate: {rate:.1f} Hz | Count: {send_count}")
        
        # Sleep for 10ms (100Hz)
        time.sleep(0.01)
    
    # Final stop - send multiple times to ensure it's received
    for _ in range(10):
        bus.send(stop_msg)
        time.sleep(0.01)
    
    total_time = time.time() - start_time
    print(f"\n[{total_time:5.1f}s] Final stop sent")
    
    print(f"\n=== SEND RESULTS ===")
    print(f"Total messages sent: {send_count}")
    print(f"Average rate: {send_count/total_time:.1f} Hz")
    print(f"Expected at 100Hz: {int(duration * 100)} messages")
    
    return send_count

def receive_phase(bus, duration=10):
    """Receive phase - receive messages for specified duration"""
    print("\n=== RECEIVE PHASE ===")
    print(f"Duration: {duration} seconds")
    print("Listening for CAN messages...\n")
    
    start_time = time.time()
    message_count = 0
    message_ids = {}
    last_print_time = 0
    
    while (time.time() - start_time) < duration:
        elapsed = time.time() - start_time
        
        # Receive with very short timeout for high-speed reception
        msg = bus.recv(timeout=0.001)
        
        if msg:
            message_count += 1
            
            # Count by ID
            if msg.arbitration_id not in message_ids:
                message_ids[msg.arbitration_id] = 0
            message_ids[msg.arbitration_id] += 1
            
            # Print summary every second instead of every message
            if int(elapsed) > last_print_time:
                last_print_time = int(elapsed)
                rate = message_count / max(0.1, elapsed)
                print(f"[{elapsed:5.1f}s] Receiving... Rate: {rate:.1f} Hz | Total: {message_count}")
                
                # Show last message details
                if msg.arbitration_id == 0x182 and len(msg.data) == 8:
                    vel, steer = struct.unpack('<ff', msg.data)
                    print(f"         Last feedback: {vel:6.2f} m/s, {steer:6.2f}°")
    
    total_time = time.time() - start_time
    
    print(f"\n=== RECEIVE RESULTS ===")
    print(f"Total messages received: {message_count}")
    if message_count > 0:
        print(f"Average rate: {message_count/total_time:.1f} Hz")
        print(f"Message breakdown by ID:")
        for can_id, count in sorted(message_ids.items()):
            print(f"  ID 0x{can_id:03X}: {count} messages ({count/total_time:.1f} Hz)")
    else:
        print("No messages received")
    
    return message_count

def send_receive_phase(bus, duration=10, velocity=2.0, steering=40.0):
    """Combined send and receive phase using threading"""
    print("\n=== SEND & RECEIVE PHASE ===")
    print(f"Duration: {duration} seconds")
    print(f"Send rate: 100Hz")
    print("Sending commands while receiving...\n")
    
    # Shared variables
    stats = {
        'send_count': 0,
        'recv_count': 0,
        'message_ids': {},
        'running': True,
        'last_feedback': None
    }
    stats_lock = threading.Lock()
    
    # Create messages
    move_msg = create_command(velocity, steering)
    stop_msg = create_command(0.0, 0.01)
    
    def send_thread():
        """Thread for sending at 100Hz"""
        start = time.time()
        while stats['running']:
            elapsed = time.time() - start
            
            # Alternate between move and stop every 2 seconds
            if int(elapsed) % 4 < 2:
                bus.send(move_msg)
            else:
                bus.send(stop_msg)
            
            with stats_lock:
                stats['send_count'] += 1
            
            time.sleep(0.01)  # 100Hz
    
    def receive_thread():
        """Thread for receiving"""
        while stats['running']:
            msg = bus.recv(timeout=0.001)
            
            if msg:
                with stats_lock:
                    stats['recv_count'] += 1
                    
                    if msg.arbitration_id not in stats['message_ids']:
                        stats['message_ids'][msg.arbitration_id] = 0
                    stats['message_ids'][msg.arbitration_id] += 1
                    
                    if msg.arbitration_id == 0x182 and len(msg.data) == 8:
                        vel, steer = struct.unpack('<ff', msg.data)
                        stats['last_feedback'] = (vel, steer)
    
    # Start threads
    send_t = threading.Thread(target=send_thread, daemon=True)
    recv_t = threading.Thread(target=receive_thread, daemon=True)
    
    send_t.start()
    recv_t.start()
    
    # Monitor progress
    start_time = time.time()
    last_print_time = 0
    
    while (time.time() - start_time) < duration:
        elapsed = time.time() - start_time
        
        # Print status every second
        if int(elapsed) > last_print_time:
            last_print_time = int(elapsed)
            
            with stats_lock:
                send_rate = stats['send_count'] / max(0.1, elapsed)
                recv_rate = stats['recv_count'] / max(0.1, elapsed)
                
                print(f"[{elapsed:5.1f}s] Send: {send_rate:.1f} Hz ({stats['send_count']}) | "
                      f"Recv: {recv_rate:.1f} Hz ({stats['recv_count']})")
                
                if stats['last_feedback']:
                    vel, steer = stats['last_feedback']
                    print(f"         Last feedback: {vel:6.2f} m/s, {steer:6.2f}°")
        
        time.sleep(0.1)
    
    # Stop threads
    stats['running'] = False
    send_t.join(timeout=0.5)
    recv_t.join(timeout=0.5)
    
    # Send final stop
    for _ in range(10):
        bus.send(stop_msg)
        time.sleep(0.01)
    
    total_time = time.time() - start_time
    
    print(f"\n=== SEND & RECEIVE RESULTS ===")
    with stats_lock:
        print(f"Messages sent: {stats['send_count']} ({stats['send_count']/total_time:.1f} Hz)")
        print(f"Messages received: {stats['recv_count']} ({stats['recv_count']/total_time:.1f} Hz)")
        if stats['recv_count'] > 0:
            print(f"Received message breakdown:")
            for can_id, count in sorted(stats['message_ids'].items()):
                print(f"  ID 0x{can_id:03X}: {count} messages ({count/total_time:.1f} Hz)")
    
    return stats['send_count'], stats['recv_count']

def main():
    parser = argparse.ArgumentParser(description='CAN Test with Phase Options (100Hz)')
    parser.add_argument('phase', choices=['send', 'receive', 'both'],
                        help='Test phase: send, receive, or both')
    parser.add_argument('-d', '--duration', type=int, default=10,
                        help='Duration in seconds (default: 10)')
    parser.add_argument('-p', '--port', default='/dev/ttyACM0',
                        help='Serial port (default: /dev/ttyACM0)')
    parser.add_argument('-v', '--velocity', type=float, default=2.0,
                        help='Velocity for send phase (default: 2.0)')
    parser.add_argument('-s', '--steering', type=float, default=40.0,
                        help='Steering for send phase (default: 40.0)')
    
    args = parser.parse_args()
    
    print("=== CAN Phase Test (100Hz) ===")
    print(f"Phase: {args.phase.upper()}")
    print(f"Duration: {args.duration} seconds")
    print(f"Port: {args.port}")
    
    try:
        # Connect to CAN
        bus = connect_can(args.port)
        
        # Run selected phase
        if args.phase == 'send':
            send_phase(bus, args.duration, args.velocity, args.steering)
        elif args.phase == 'receive':
            receive_phase(bus, args.duration)
        elif args.phase == 'both':
            send_receive_phase(bus, args.duration, args.velocity, args.steering)
        
        print("\n✓ Test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        # Send stop command
        try:
            stop_msg = create_command(0.0, 0.0)
            for _ in range(10):
                bus.send(stop_msg)
                time.sleep(0.01)
            print("✓ Emergency stop sent")
        except:
            pass
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check device connection: lsusb | grep 16d0")
        print("2. Check permissions: ls -la /dev/ttyACM*")
        print("3. Try: sudo chmod 666 /dev/ttyACM0")
        return 1
        
    finally:
        if 'bus' in locals():
            bus.shutdown()
            print("✓ Connection closed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())