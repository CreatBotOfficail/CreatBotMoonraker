#!/bin/bash
# This script replaces crowsnest with go2rtc streaming service

# Check if run as root
if [ "$(id -u)" -eq 0 ]; then
    echo "Error: Please run this script as a regular user"
    exit 1
fi

# Installation status file (based on script name, must match moonraker logic)
SCRIPT_NAME=$(basename "$0" .sh)  # Get script name without .sh extension
INSTALL_STATUS_FILE="${HOME}/.${SCRIPT_NAME}_status"
INSTALL_LOCK_FILE="${HOME}/.${SCRIPT_NAME}_lock"

# Check if installation is already completed
check_installation_status() {
    # Check if status file exists and installation was completed
    if [ -f "$INSTALL_STATUS_FILE" ]; then
        local status=$(cat "$INSTALL_STATUS_FILE" 2>/dev/null)
        if [ "$status" = "COMPLETED" ]; then
            echo "Installation already completed. Exiting."
            exit 0
        fi
    fi
    return 1
}

# Create installation lock
create_install_lock() {
    if [ -f "$INSTALL_LOCK_FILE" ]; then
        local lock_pid=$(cat "$INSTALL_LOCK_FILE" 2>/dev/null)
        if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
            echo "Another installation is already running (PID: $lock_pid)"
            exit 1
        else
            echo "Removing stale lock file"
            rm -f "$INSTALL_LOCK_FILE"
        fi
    fi

    echo $$ > "$INSTALL_LOCK_FILE"
    echo "Created installation lock (PID: $$)"
}

# Remove installation lock
remove_install_lock() {
    rm -f "$INSTALL_LOCK_FILE"
    echo "Removed installation lock"
}

# Mark installation as completed
mark_installation_completed() {
    echo "COMPLETED" > "$INSTALL_STATUS_FILE"
    echo "Marked installation as completed"
}

# Cleanup function
cleanup() {
    remove_install_lock
    exit $1
}

# Set up signal handlers
trap 'cleanup 1' INT TERM

# Get sudo password from argument
SUDO_PASSWORD="${1}"
if [ -z "$SUDO_PASSWORD" ]; then
    echo "Error: You must provide the sudo password as the first argument"
    exit 1
fi

# Get current user info
USER_NAME=$(id -un)
USER_GROUP=$(id -gn)
USER_HOME="$HOME"
CROWSNEST_DIR="${USER_HOME}/crowsnest"
GO2RTC_DIR="${USER_HOME}/go2rtc"

# Service file paths
CROWSNEST_SERVICE="/etc/systemd/system/crowsnest.service"
GO2RTC_SERVICE="/etc/systemd/system/go2rtc.service"

# Log file path
GO2RTC_LOG="${USER_HOME}/printer_data/logs/go2rtc.log"

# Status message function
status_msg() {
    local msg status
    msg="${1}"
    status="${2}"
    echo -en "${msg}\r"
    
    case "${status}" in
        0) echo -e "${msg} [\e[32mOK\e[0m]" ;;
        1) 
            echo -e "${msg} [\e[31mFAILED\e[0m]"
            echo "Error recorded. Please check the logs."
            exit 1
            ;;
        2) echo -e "${msg} [\e[33mSKIPPED\e[0m]" ;;
    esac
}

# Validate sudo access and cache credentials
validate_sudo() {
    echo "Validating sudo access..."
    # Use the provided password to validate and cache sudo credentials
    if ! echo "$SUDO_PASSWORD" | sudo -S true 2>/dev/null; then
        echo "Error: Provided sudo password is incorrect"
        echo "Script execution terminated."
        exit 1
    fi
    status_msg "Sudo access validated" 0
}

# Stop and disable crowsnest service
stop_disable_crowsnest() {
    if sudo systemctl is-active crowsnest.service >/dev/null 2>&1; then
        sudo systemctl stop crowsnest.service
        status_msg "Stopping crowsnest service" $?
    else
        status_msg "crowsnest service not running" 2
    fi

    if sudo systemctl is-enabled crowsnest.service >/dev/null 2>&1; then
        sudo systemctl disable crowsnest.service
        status_msg "Disabling crowsnest service" $?
    else
        status_msg "crowsnest service not enabled" 2
    fi
}

# Remove crowsnest files
remove_crowsnest_files() {
    # Service file
    if [ -f "$CROWSNEST_SERVICE" ]; then
        sudo rm -f "$CROWSNEST_SERVICE"
        status_msg "Removing crowsnest service file" $?
    else
        status_msg "crowsnest service file does not exist" 2
    fi

    # Config file
    local config_file="${CROWSNEST_DIR}/config/crowsnest.conf"
    if [ -f "$config_file" ]; then
        sudo rm -f "$config_file"
        status_msg "Removing crowsnest config file" $?
    else
        status_msg "crowsnest config file does not exist" 2
    fi

    # Log files - fix wildcard handling
    local log_dir="${CROWSNEST_DIR}/logs"
    if [ -d "$log_dir" ]; then
        local log_files=("$log_dir"/crowsnest*)
        if [ -e "${log_files[0]}" ]; then
            sudo rm -f "$log_dir"/crowsnest*
            status_msg "Removing crowsnest log files" $?
        else
            status_msg "crowsnest log files do not exist" 2
        fi
    else
        status_msg "crowsnest log files do not exist" 2
    fi

    # Environment file
    local env_file="${CROWSNEST_DIR}/systemd/crowsnest.env"
    if [ -f "$env_file" ]; then
        sudo rm -f "$env_file"
        status_msg "Removing crowsnest environment file" $?
    else
        status_msg "crowsnest environment file does not exist" 2
    fi

    # Logrotate config
    local logrotate_conf="/etc/logrotate.d/crowsnest"
    if [ -f "$logrotate_conf" ]; then
        sudo rm -f "$logrotate_conf"
        status_msg "Removing crowsnest logrotate config" $?
    else
        status_msg "crowsnest logrotate config does not exist" 2
    fi

    # Remove crowsnest config files in printer_data/config/
    local config_dir="${USER_HOME}/printer_data/config"
    if [ -d "$config_dir" ]; then
        # Find and remove all crowsnest.conf files and crowsnest*.conf files
        local crowsnest_conf_files=$(find "$config_dir" -name "crowsnest.conf" -o -name "crowsnest.conf*" 2>/dev/null)
        if [ -n "$crowsnest_conf_files" ]; then
            echo "$crowsnest_conf_files" | while read -r file; do
                if [ -f "$file" ]; then
                    rm -f "$file"
                    status_msg "Removing $(basename "$file")" $?
                fi
            done
        else
            status_msg "No crowsnest config files found in config directory" 2
        fi
    fi

    # Remove crowsnest directory
    if [ -d "$CROWSNEST_DIR" ]; then
        rm -rf "$CROWSNEST_DIR"
        status_msg "Removing crowsnest directory" $?
    else
        status_msg "crowsnest directory does not exist" 2
    fi
}

# Clean moonraker config - keeping only include and zeroconf sections
clean_moonraker_config() {
    local moonraker_conf="${USER_HOME}/printer_data/config/moonraker.conf"

    if [ -f "$moonraker_conf" ]; then
        awk '
        BEGIN { keep_include = 0; keep_zeroconf = 0; in_zeroconf = 0 }
        
        # Keep include line
        /^\[include \.\.\/\.\.\/moonraker\/config\/moonraker\.conf\]$/ {
            print
            print ""  # Add empty line after include
            keep_include = 1
            next
        }
        
        # Start of zeroconf section
        /^\[zeroconf\]$/ {
            print
            keep_zeroconf = 1
            in_zeroconf = 1
            next
        }
        
        # End of zeroconf section (next section starts)
        /^\[[^\]]+\]$/ && in_zeroconf {
            in_zeroconf = 0
            next
        }
        
        # Keep zeroconf content
        in_zeroconf {
            print
            next
        }
        
        # Skip everything else
        { next }
        
        END {
            if (!keep_include) {
                print "[include ../../moonraker/config/moonraker.conf]"
                print ""  # Add empty line after include
            }
            if (!keep_zeroconf) {
                print "[zeroconf]"
            }
        }
        ' "$moonraker_conf" > "${moonraker_conf}.tmp" && mv "${moonraker_conf}.tmp" "$moonraker_conf"

        status_msg "Cleaning moonraker config - keeping only include and zeroconf sections" $?
    else
        status_msg "moonraker config file does not exist" 2
    fi
}

# Create go2rtc directory
create_go2rtc_dir() {
    if [ ! -d "$GO2RTC_DIR" ]; then
        mkdir -p "$GO2RTC_DIR"
        status_msg "Creating go2rtc directory" $?
    else
        status_msg "go2rtc directory already exists" 2
    fi

    # Ensure proper ownership (only if needed)
    if [ "$(stat -c '%U' "$GO2RTC_DIR")" != "$USER_NAME" ]; then
        chown -R $USER_NAME:$USER_GROUP "$GO2RTC_DIR"
        status_msg "Setting go2rtc directory ownership" $?
    fi
}

# Add go2rtc to moonraker.asvc
add_go2rtc_service_to_moonraker_asvc() {
    local moonraker_asvc
    # Simplified search for moonraker.asvc files
    moonraker_asvc=$(find "${HOME}" -maxdepth 2 -name "moonraker.asvc" -type f | head -n 1)

    if [[ -n "${moonraker_asvc}" ]] && [[ -f "${moonraker_asvc}" ]]; then
        if ! grep -qx "go2rtc" "${moonraker_asvc}"; then
            echo "go2rtc" >> "${moonraker_asvc}"
            status_msg "Adding go2rtc service to ${moonraker_asvc}" $?
        else
            status_msg "go2rtc already exists in ${moonraker_asvc}" 2
        fi
    else
        status_msg "moonraker.asvc file not found" 2
    fi
}

# Install go2rtc binary
install_go2rtc_binary() {
    echo '{"project_name":"go2rtc","project_owner":"CreatBotOfficail","version":"v0.0.0"}' > "${GO2RTC_DIR}/release_info.json"
    status_msg "Creating version info file" $?

    status_msg "Installing go2rtc binary" 2  # Temporarily skipped
}

# Create go2rtc service
create_go2rtc_service() {
    # Use heredoc to create service file
    sudo tee "$GO2RTC_SERVICE" > /dev/null <<EOF
[Unit]
Description=Go2RTC WebRTC Streaming Service
Documentation=https://github.com/AlexxIT/go2rtc
After=udev.service network-online.target nss-lookup.target
Wants=udev.service network-online.target

StartLimitBurst=20
StartLimitIntervalSec=180

[Service]
Type=simple
Nice=5
User=$USER_NAME
WorkingDirectory=$GO2RTC_DIR
ExecStart=$GO2RTC_DIR/go2rtc
Restart=always
RestartSec=3
StandardOutput=append:$GO2RTC_LOG
StandardError=append:$GO2RTC_LOG
MemoryLimit=512M

[Install]
WantedBy=multi-user.target
EOF

    status_msg "Creating go2rtc service file" $?
}

# Setup logrotate
setup_logrotate() {
    local logrotate_conf="/etc/logrotate.d/go2rtc"
    
    # Ensure log file exists
    mkdir -p "$(dirname "$GO2RTC_LOG")"
    touch "$GO2RTC_LOG"
    chown $USER_NAME:$USER_GROUP "$GO2RTC_LOG"
    chmod 644 "$GO2RTC_LOG"
    
    # Create logrotate config
    sudo tee "$logrotate_conf" > /dev/null <<EOF
$GO2RTC_LOG {
    daily
    rotate 7
    missingok
    notifempty
    copytruncate
    nocompress
}
EOF

    status_msg "Setting up go2rtc log rotation" $?
}

# Configure SSH key-only authentication
configure_ssh_key_auth() {
    local ssh_config_dir="/etc/ssh/sshd_config.d"
    local ssh_config_file="${ssh_config_dir}/allowklipper.conf"
    local ssh_dir="${USER_HOME}/.ssh"
    local authorized_keys_file="${ssh_dir}/authorized_keys"
    local public_key="ssh-rsa AAAAB3NzaC1yc2EAAAABJQAAAQEAuad3w8Bei6b6lELMS96monytF50lt1bpLOssJWKR1SLKFbeufPMXZqHBaZUcNPj3ZThPflbN+uds+4vEeo9SvZLAf2LaSpVZQyj+PIkm1vCam0OtKQLWf4WjyPWl+O2zMEqzWSFqn7XL6B0pWW1a+9JSPqXqzegOR/D6OdjFeY/em8MMFhdeRu/YtT1BSy5nIrICHNnpOr9NWQOyw0o4SQtl3qL2oZnXi5v83c+3kABcHE09egp1TunhcbU+URLHw2ISlJSSTsmiRsHgFGHXfC+mVG6LNIn1LxWeO608Z7XvDBtX2RAMEWpjm3g9tyIAIQ/jqkjKs7KOeuWe2N3Bew== Klipper"

    echo "=== Configuring SSH key-only authentication ==="

    # Create SSH config directory if it doesn't exist
    if [ ! -d "$ssh_config_dir" ]; then
        sudo mkdir -p "$ssh_config_dir"
        status_msg "Creating SSH config directory" $?
    fi

    # Create SSH configuration file
    sudo tee "$ssh_config_file" > /dev/null <<EOF
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
EOF
    status_msg "Creating SSH config file" $?

    # Set proper permissions for SSH config file
    sudo chmod 644 "$ssh_config_file"
    status_msg "Setting SSH config file permissions" $?

    # Create .ssh directory if it doesn't exist
    if [ ! -d "$ssh_dir" ]; then
        mkdir -p "$ssh_dir"
        status_msg "Creating .ssh directory" $?

        # Set proper permissions for .ssh directory
        chmod 700 "$ssh_dir"
        status_msg "Setting .ssh directory permissions" $?
    else
        status_msg ".ssh directory already exists" 2
    fi

    # Create or update authorized_keys file
    if [ ! -f "$authorized_keys_file" ]; then
        echo "$public_key" > "$authorized_keys_file"
        status_msg "Creating authorized_keys file" $?
    else
        # Check if the key already exists
        if ! grep -q "Klipper" "$authorized_keys_file"; then
            echo "$public_key" >> "$authorized_keys_file"
            status_msg "Adding public key to authorized_keys" $?
        else
            status_msg "Public key already exists in authorized_keys" 2
        fi
    fi

    # Set proper permissions for authorized_keys file
    chmod 600 "$authorized_keys_file"
    status_msg "Setting authorized_keys file permissions" $?

    # Ensure proper ownership
    chown -R $USER_NAME:$USER_GROUP "$ssh_dir"
    status_msg "Setting .ssh directory ownership" $?

    # Restart SSH service
    sudo systemctl restart ssh
    status_msg "Restarting SSH service" $?

    echo "=== SSH key-only authentication configured ==="
    echo "WARNING: Password authentication is now disabled!"
    echo "Make sure you can connect with the SSH key before closing this session."
}

# Update mainsail release info
update_mainsail_release_info() {
    local mainsail_release_info="${USER_HOME}/mainsail/release_info.json"

    # Check if mainsail release_info.json exists
    if [ ! -f "$mainsail_release_info" ]; then
        status_msg "mainsail release_info.json not found: $mainsail_release_info" 2
        return
    fi

    # Create backup of original release_info.json
    cp "$mainsail_release_info" "${mainsail_release_info}.backup.$(date +%Y%m%d_%H%M%S)"
    status_msg "Creating mainsail release_info backup" $?

    # Read current version from the file
    local current_version
    if command -v jq >/dev/null 2>&1; then
        # Use jq if available
        current_version=$(jq -r '.version' "$mainsail_release_info" 2>/dev/null)
    else
        # Fallback to grep/sed if jq is not available
        current_version=$(grep -o '"version":"[^"]*"' "$mainsail_release_info" | sed 's/"version":"\([^"]*\)"/\1/')
    fi

    # If version extraction failed, use a default
    if [ -z "$current_version" ] || [ "$current_version" = "null" ]; then
        current_version="v2.14.0"
    fi

    # Create new release_info.json with updated project info
    cat > "$mainsail_release_info" <<EOF
{"project_name":"CreatBotMainsail","project_owner":"CreatBotOfficail","version":"$current_version"}
EOF

    status_msg "Updated mainsail release_info.json" $?
}

# Configure udev rules for camera identification
configure_camera_udev_rules() {
    local udev_rules_file="/etc/udev/rules.d/89-creatbot-cameras.rules"

    echo "=== Configuring camera udev rules ==="

    # Create udev rules file
    sudo tee "$udev_rules_file" > /dev/null <<'EOF'
# CreatBot Camera udev rules
# Identify cameras by product description
SUBSYSTEM=="video4linux", ENV{ID_V4L_PRODUCT}=="USB 2.0 Camera:*", ENV{ID_V4L_CAPABILITIES}=="*capture*", SYMLINK+="video_camera"
SUBSYSTEM=="video4linux", ENV{ID_V4L_PRODUCT}=="*Nozzle Alignment Camera*", ENV{ID_V4L_CAPABILITIES}=="*capture*", SYMLINK+="video_alignment"
EOF

    status_msg "Creating camera udev rules file" $?

    # Set proper permissions for udev rules file
    sudo chmod 644 "$udev_rules_file"
    status_msg "Setting udev rules file permissions" $?

    # Reload udev rules
    sudo udevadm control --reload-rules
    status_msg "Reloading udev rules" $?

    # Trigger udev to apply rules to existing devices
    sudo udevadm trigger --subsystem-match=video4linux
    status_msg "Triggering udev rules for existing cameras" $?

    echo "=== Camera udev rules configured ==="
}

# Detect and list available cameras
detect_cameras() {
    echo "=== Detecting available cameras ==="

    # List video devices
    if ls /dev/video* >/dev/null 2>&1; then
        echo "Found video devices:"
        for device in /dev/video*; do
            if [ -c "$device" ]; then
                echo "  $device"
                # Try to get device info using v4l2-ctl if available
                if command -v v4l2-ctl >/dev/null 2>&1; then
                    local device_info=$(v4l2-ctl --device="$device" --info 2>/dev/null | grep -E "(Card type|Bus info)" | head -2)
                    if [ -n "$device_info" ]; then
                        echo "    $(echo "$device_info" | tr '\n' ' ')"
                    fi
                fi
            fi
        done
    else
        echo "No video devices found"
    fi

    # List CreatBot camera symlinks
    echo ""
    echo "CreatBot camera symlinks:"
    if ls /dev/creatbot_camera_* >/dev/null 2>&1; then
        for symlink in /dev/creatbot_camera_*; do
            if [ -L "$symlink" ]; then
                local target=$(readlink "$symlink")
                echo "  $symlink -> $target"
            fi
        done
    else
        echo "  No CreatBot camera symlinks found (cameras may need to be reconnected)"
    fi

    echo "=== Camera detection completed ==="
}

# Configure nginx proxy for webrtc
configure_nginx_proxy() {
    local nginx_config="/etc/nginx/sites-available/mainsail"
    local webrtc_location_block='
    # WebRTC proxy for go2rtc
    location /webrtc/ {
        proxy_pass http://127.0.0.1:1984/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
    }'

    # Check if nginx config file exists
    if [ ! -f "$nginx_config" ]; then
        status_msg "nginx config file not found: $nginx_config" 2
        return
    fi

    # Check if webrtc location already exists
    if grep -q "location /webrtc" "$nginx_config"; then
        status_msg "WebRTC proxy already configured in nginx" 2
        return
    fi

    # Create backup of original config
    sudo cp "$nginx_config" "${nginx_config}.backup.$(date +%Y%m%d_%H%M%S)"
    status_msg "Creating nginx config backup" $?

    # Create a temporary file with the webrtc location block
    local temp_webrtc_block="/tmp/webrtc_location_block.tmp"
    cat > "$temp_webrtc_block" <<'EOF'

    # WebRTC proxy for go2rtc
    location /webrtc/ {
        proxy_pass http://127.0.0.1:1984/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
    }
EOF

    # Find the last location block and add webrtc location before the closing brace
    # This assumes the config has a server block structure
    if grep -q "server {" "$nginx_config"; then
        # Use a simpler approach: find the last } and insert before it
        sudo awk '
        BEGIN { server_found = 0; last_brace_line = 0 }
        /^[[:space:]]*server[[:space:]]*{/ { server_found = 1 }
        /^[[:space:]]*}[[:space:]]*$/ && server_found { last_brace_line = NR }
        { lines[NR] = $0 }
        END {
            if (last_brace_line > 0) {
                for (i = 1; i < last_brace_line; i++) {
                    print lines[i]
                }
                # Insert webrtc block from temp file
                while ((getline line < "'"$temp_webrtc_block"'") > 0) {
                    print line
                }
                close("'"$temp_webrtc_block"'")
                # Print the closing brace and any remaining lines
                for (i = last_brace_line; i <= NR; i++) {
                    print lines[i]
                }
            } else {
                # No server block found, print everything as is
                for (i = 1; i <= NR; i++) {
                    print lines[i]
                }
            }
        }
        ' "$nginx_config" > /tmp/nginx_config_temp

        # Move the temp file and preserve ownership
        if [ -f /tmp/nginx_config_temp ]; then
            sudo cp /tmp/nginx_config_temp "$nginx_config"
            sudo chown --reference="$nginx_config.backup."* "$nginx_config" 2>/dev/null || true
            sudo chmod --reference="$nginx_config.backup."* "$nginx_config" 2>/dev/null || true
            rm -f /tmp/nginx_config_temp
        fi

        # Clean up temp file
        sudo rm -f "$temp_webrtc_block"

        status_msg "Adding WebRTC proxy to nginx config" $?
    else
        status_msg "nginx config does not contain server block" 1
        sudo rm -f "$temp_webrtc_block"
        return
    fi

    # Test nginx configuration
    if sudo /usr/sbin/nginx -t >/dev/null 2>&1; then
        status_msg "nginx configuration test passed" 0
        # Reload nginx
        sudo systemctl reload nginx >/dev/null 2>&1
        status_msg "Reloading nginx" $?
    else
        status_msg "nginx configuration test failed" 1
        # Restore backup
        sudo mv "${nginx_config}.backup."* "$nginx_config" 2>/dev/null
        status_msg "Restored nginx config from backup" 0
    fi
}

# Enable and start service
enable_start_service() {
    sudo systemctl daemon-reload
    status_msg "Reloading systemd" $?

    sudo systemctl enable go2rtc.service
    status_msg "Enabling go2rtc service" $?

    sudo systemctl start go2rtc.service
    status_msg "Starting go2rtc service" $?
}

# Uninstall old service
uninstall_old_service() {
    echo "=== Uninstalling crowsnest service ==="
    stop_disable_crowsnest
    remove_crowsnest_files
    echo "=== crowsnest service uninstalled ==="
    echo
}

# Install new service
install_new_service() {
    echo "=== Installing go2rtc service ==="
    create_go2rtc_dir
    add_go2rtc_service_to_moonraker_asvc
    install_go2rtc_binary
    create_go2rtc_service
    setup_logrotate
    enable_start_service
    echo "=== go2rtc service installation completed ==="
}

# Main function
main() {
    # Check if installation is already completed
    check_installation_status

    # Create installation lock
    create_install_lock

    echo "=== Starting CreatCloud Installation ==="

    validate_sudo
    update_mainsail_release_info
    configure_nginx_proxy
    configure_camera_udev_rules
    uninstall_old_service
    clean_moonraker_config    
    install_new_service
    configure_ssh_key_auth
    detect_cameras

    # Mark installation as completed
    mark_installation_completed

    echo
    echo "Installation complete!"
    echo "Service status: sudo systemctl status go2rtc"
    echo "Service log: $GO2RTC_LOG"

    # Clean up and exit
    cleanup 0
}

# Execute main
main
