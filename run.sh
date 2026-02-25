#!/bin/bash

#############################
# PVP4Real Docker Manager
#############################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROS_DOMAIN_ID=25
export DISPLAY=localhost:10.0

#############################
# TUI Helper Functions
#############################

# Check if dialog is available, fallback to whiptail
check_tui_tool() {
  if command -v dialog &> /dev/null; then
    echo "dialog"
  elif command -v whiptail &> /dev/null; then
    echo "whiptail"
  else
    echo "none"
  fi
}

# Display message box
show_message() {
  local tui_tool=$1
  local title=$2
  local message=$3
  local height=${4:-10}
  local width=${5:-60}
  
  if [ "$tui_tool" = "dialog" ]; then
    dialog --title "$title" --msgbox "$message" $height $width
  elif [ "$tui_tool" = "whiptail" ]; then
    whiptail --title "$title" --msgbox "$message" $height $width
  else
    echo "$title"
    echo "$message"
    read -p "Press Enter to continue..."
  fi
}

# Display error and exit
show_error() {
  local tui_tool=$1
  local message=$2
  
  if [ "$tui_tool" = "dialog" ]; then
    dialog --title "Error" --msgbox "$message" 10 60
    clear
  elif [ "$tui_tool" = "whiptail" ]; then
    whiptail --title "Error" --msgbox "$message" 10 60
  else
    echo "Error: $message"
  fi
  exit 1
}

# TUI menu for selecting action
select_action_tui() {
  local tui_tool=$1
  
  local menu_options=()
  menu_options+=("dev" "Development (interactive shell)")
  menu_options+=("online" "Online Training (Human-in-the-Loop)")
  menu_options+=("deploy" "Deployment (application runtime)")
  menu_options+=("record" "Record ROS2 bag")
  menu_options+=("cb" "Build colcon workspace")
  menu_options+=("────────── " "")
  menu_options+=("build" "Build Docker image")
  
  local selection
  if [ "$tui_tool" = "dialog" ]; then
    selection=$(DIALOGRC=/dev/null dialog --colors \
      --backtitle "PVP4Real Docker Manager" \
      --title " Select Action " \
      --cancel-label "Quit" \
      --menu "\nWhat would you like to do?" 15 60 5 \
      "${menu_options[@]}" \
      2>&1 >/dev/tty)
  elif [ "$tui_tool" = "whiptail" ]; then
    selection=$(whiptail --clear \
      --backtitle "PVP4Real Docker Manager" \
      --title " Select Action " \
      --cancel-button "Quit" \
      --menu "\nWhat would you like to do?" 15 60 5 \
      "${menu_options[@]}" \
      3>&1 1>&2 2>&3)
  fi
  
  local exit_code=$?
  
  # Check if separator was selected
  if [ "$selection" = "────────── " ]; then
    return 3  # Separator selected, stay on menu
  fi
  
  if [ $exit_code -ne 0 ]; then
    clear
    return 1  # Quit
  fi
  
  echo "$selection"
}

# Show summary before starting
show_summary_tui() {
  local tui_tool=$1
  local action=$2
  
  local message="Configuration Summary:\n\n"
  message+="Action: $action\n"
  message+="Docker Image: pomelo925/pvp4real:latest\n"
  message+="Compose File: docker/compose.cpu.yml\n"
  message+="Workspace: workspace/ -> /workspace\n\n"
  message+="Press OK to continue..."
  
  show_message "$tui_tool" "Confirmation" "$message" 14 60
}

#############################
# Command-line Usage
#############################

usage() {
  cat << EOF
Usage: $0 [action]

action:
  build              Build Docker image
  dev                Development service (interactive shell with volumes)
  online             Online Training / HITL mode (runs pvp.hitl.py)
  deploy             Deploy service (runs pvp.deploy.py)
  record             Record ROS2 bag (runs rosbag.py)
  cb                 Build ROS2 workspace with colcon (auto-exit after build)

Examples:
  $0 build cpu       # Build PVP4Real CPU Docker image
  $0 dev gpu-cu124   # Start PVP4Real GPU development environment
  $0 online cpu      # Start PVP4Real Online Training / HITL mode (CPU)
  $0 deploy gpu-cu124 # Start PVP4Real GPU deployment service
  $0 record cpu      # Record ROS2 bag
  $0 cb cpu          # Build ROS2 workspace and exit
  $0                 # Interactive TUI menu mode

Configuration:
  Docker files: docker/dockerfile.cpu, docker/dockerfile.gpu.cu124
  Compose files: docker/compose.cpu.yml, docker/compose.gpu.cu124.yml
  Services: dev, online, deploy, record, cb
  Workspace: workspace/ (mounted to /workspace in container)
EOF
  exit 1
}

# Setup X11 forwarding for GUI applications
setup_x11_forwarding() {
  echo "Setting up X11 forwarding..."
  xhost +local:docker > /dev/null 2>&1 || true
}

# Build docker image
build_docker_image() {
  local compose_file=$1
  
  echo "=========================================="
  echo "Building Docker image..."
  echo "Compose file: $compose_file"
  echo "=========================================="
  
  docker compose -f "$compose_file" build
  
  if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Build completed successfully!"
    echo "=========================================="
  else
    echo ""
    echo "=========================================="
    echo "Build failed!"
    echo "=========================================="
    exit 1
  fi
}

# Cleanup existing containers
cleanup_containers() {
  local project_name=$1
  local compose_file=$2
  
  echo "Cleaning up existing containers..."
  docker compose -p "$project_name" -f "$compose_file" down --volumes --remove-orphans 2>/dev/null || true
}

# Start service
start_service() {
  local project_name=$1
  local compose_file=$2
  local service=$3
  
  echo "=========================================="
  echo "Starting service..."
  echo "Service: $service"
  echo "=========================================="
  
  docker compose -p "$project_name" -f "$compose_file" up -d "$service"
  
  if [ $? -ne 0 ]; then
    echo "Failed to start service!"
    exit 1
  fi
  
  echo ""
  echo "Service started successfully!"
  echo "Container logs:"
  echo "=========================================="
  sleep 2
  docker compose -p "$project_name" -f "$compose_file" logs "$service"
}

# Enter container
enter_container() {
  local project_name=$1
  local compose_file=$2
  local service=$3
  
  echo ""
  echo "=========================================="
  echo "Entering container..."
  echo "=========================================="
  
  docker compose -p "$project_name" -f "$compose_file" exec "$service" bash
}

main() {
  # Set up configuration
  compose_file="$SCRIPT_DIR/docker/compose.cpu.yml"
  project_name="pvp4real-cpu"
  
  if [ ! -f "$compose_file" ]; then
    echo "Error: Compose file not found: $compose_file"
    exit 1
  fi
  
  # Check if arguments are provided
  if [ $# -eq 0 ]; then
    # Interactive TUI mode
    tui_tool=$(check_tui_tool)
    
    if [ "$tui_tool" = "none" ]; then
      echo "Warning: Neither dialog nor whiptail found. Using command-line mode."
      echo "Install dialog or whiptail for interactive menu: sudo apt install dialog"
      echo ""
      usage
    fi
    
    # Interactive loop
    while true; do
      action=$(select_action_tui "$tui_tool")
      local action_result=$?
      
      [ $action_result -eq 1 ] && exit 0  # Quit
      [ $action_result -eq 3 ] && continue # Separator selected, stay on menu
      
      # Show summary
      show_summary_tui "$tui_tool" "$action"
      
      clear
      
      # Execute action
      if [ "$action" = "build" ]; then
        build_docker_image "$compose_file"
        read -p "Press Enter to return to menu..."
        continue
      elif [ "$action" = "cb" ]; then
        echo "=========================================="
        echo "Running colcon build..."
        echo "=========================================="
        docker compose -p "$project_name" -f "$compose_file" up cb
        echo ""
        echo "=========================================="
        echo "Colcon build completed!"
        echo "=========================================="
        read -p "Press Enter to return to menu..."
        continue
      else
        # Start interactive service
        service=$action
        setup_x11_forwarding
        cleanup_containers "$project_name" "$compose_file"
        start_service "$project_name" "$compose_file" "$service"
        enter_container "$project_name" "$compose_file" "$service"
        
        # Cleanup message
        echo ""
        echo "=========================================="
        echo "Service session ended."
        echo "To stop the service, run:"
        echo "  docker compose -p $project_name -f $compose_file down"
        echo "=========================================="
        
        exit 0
      fi
    done
  else
    # Command-line mode
    action=$1
    
    # Validate inputs
    if [ "$action" != "build" ] && [ "$action" != "dev" ] && [ "$action" != "online" ] && [ "$action" != "deploy" ] && [ "$action" != "record" ] && [ "$action" != "cb" ]; then
      echo "Error: Invalid action '$action'. Must be 'build', 'dev', 'online', 'deploy', 'record', or 'cb'."
      usage
    fi
    
    if [ "$action" = "build" ]; then
      # Build docker image
      build_docker_image "$compose_file"
    elif [ "$action" = "cb" ]; then
      # Run colcon build service
      echo "=========================================="
      echo "Running colcon build..."
      echo "=========================================="
      docker compose -p "$project_name" -f "$compose_file" up cb
      echo ""
      echo "=========================================="
      echo "Colcon build completed!"
      echo "=========================================="
    else
      # Setup and start service
      service=$action
      setup_x11_forwarding
      cleanup_containers "$project_name" "$compose_file"
      start_service "$project_name" "$compose_file" "$service"
      enter_container "$project_name" "$compose_file" "$service"
      
      # Cleanup message
      echo ""
      echo "=========================================="
      echo "Service session ended."
      echo "To stop the service, run:"
      echo "  docker compose -p $project_name -f $compose_file down"
      echo "=========================================="
    fi
  fi
}

# Run main function
main "$@"
