#!/bin/bash

#############################
# PVP4Real Docker Manager
#############################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROS_DOMAIN_ID=25
export DISPLAY=:1

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

# TUI menu for selecting platform (cpu/gpu)
select_platform_tui() {
  local tui_tool=$1
  local menu_options=("cpu" "CPU (default)" "gpu-cu124" "GPU (CUDA 12.4)")
  local selection
  if [ "$tui_tool" = "dialog" ]; then
    selection=$(DIALOGRC=/dev/null dialog --colors \
      --backtitle "PVP4Real Docker Manager" \
      --title " Select Platform " \
      --cancel-label "Back" \
      --menu "\nWhich platform do you want to use?" 12 60 2 \
      "${menu_options[@]}" \
      2>&1 >/dev/tty)
  elif [ "$tui_tool" = "whiptail" ]; then
    selection=$(whiptail --clear \
      --backtitle "PVP4Real Docker Manager" \
      --title " Select Platform " \
      --cancel-button "Back" \
      --menu "\nWhich platform do you want to use?" 12 60 2 \
      "${menu_options[@]}" \
      3>&1 1>&2 2>&3)
  fi
  local exit_code=$?
  if [ $exit_code -ne 0 ]; then
    return 1  # Back
  fi
  echo "$selection"
}

# Show summary before starting
show_summary_tui() {
  local tui_tool=$1
  local action=$2
  local compose_file=$3
  local docker_image="pomelo925/pvp4real:latest"
  if [[ "$compose_file" == *gpu* ]]; then
    docker_image="pomelo925/pvp4real:gpu-cu124"
  fi
  local message="Configuration Summary:\n\n"
  message+="Action: $action\n"
  message+="Docker Image: $docker_image\n"
  message+="Compose File: ${compose_file##*/}\n"
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
  # Interactive TUI or CLI mode
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

      # Platform selection (skip for separator)
      while true; do
        platform=$(select_platform_tui "$tui_tool")
        local platform_result=$?
        [ $platform_result -eq 1 ] && break # Back to action menu

        # Set compose_file and project_name based on platform
        if [ "$platform" = "gpu-cu124" ]; then
          compose_file="$SCRIPT_DIR/docker/compose.gpu.cu124.yml"
          project_name="pvp4real-gpu-cu124"
        else
          compose_file="$SCRIPT_DIR/docker/compose.cpu.yml"
          project_name="pvp4real-cpu"
        fi

        # Show summary
        show_summary_tui "$tui_tool" "$action" "$compose_file"
        clear

        # Execute action
        if [ "$action" = "build" ]; then
          build_docker_image "$compose_file"
          read -p "Press Enter to return to menu..."
          break
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
          break
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
    done
  else
    # Command-line mode (default to cpu if not specified)
    action=$1
    platform=${2:-cpu}
    if [ "$platform" = "gpu-cu124" ]; then
      compose_file="$SCRIPT_DIR/docker/compose.gpu.cu124.yml"
      project_name="pvp4real-gpu-cu124"
    else
      compose_file="$SCRIPT_DIR/docker/compose.cpu.yml"
      project_name="pvp4real-cpu"
    fi
    # Validate inputs
    if [ "$action" != "build" ] && [ "$action" != "dev" ] && [ "$action" != "online" ] && [ "$action" != "deploy" ] && [ "$action" != "record" ] && [ "$action" != "cb" ]; then
      echo "Error: Invalid action '$action'. Must be 'build', 'dev', 'online', 'deploy', 'record', or 'cb'."
      usage
    fi
    if [ "$action" = "build" ]; then
      build_docker_image "$compose_file"
    elif [ "$action" = "cb" ]; then
      echo "=========================================="
      echo "Running colcon build..."
      echo "=========================================="
      docker compose -p "$project_name" -f "$compose_file" up cb
      echo ""
      echo "=========================================="
      echo "Colcon build completed!"
      echo "=========================================="
    else
      service=$action
      setup_x11_forwarding
      cleanup_containers "$project_name" "$compose_file"
      start_service "$project_name" "$compose_file" "$service"
      enter_container "$project_name" "$compose_file" "$service"
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
