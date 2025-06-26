#!/bin/bash

#
#   ejabberd container installer for macOS/Linux
#   ------------------------------------------
#                                       v0.4
#
# This shell script downloads an ejabberd container image
# and sets up a docker container to run ejabberd.

#
# 1. Install Docker:
#
#    For macOS: Download Docker Desktop from https://www.docker.com/
#    For Linux: Install docker and docker-compose using your package manager
#               Example for Ubuntu/Debian: sudo apt install docker.io docker-compose
#               Example for CentOS/RHEL: sudo yum install docker docker-compose
#

#
# 2. Edit these options:

# Directory where your ejabberd deployment files will be installed
# (configuration, database, logs, ...)
INSTALL_DIR="$HOME/ejabberd"

# Please enter the desired ejabberd domain name.
# The domain is the visible attribute that is added to the username
# to form the Jabber Identifier (for example: user@example.net).
# This computer must be known on the network with this address name.
# You can later add more in conf/ejabberd.yml
HOST="localhost"

# Please enter the administrator username for the current
# ejabberd installation. A Jabber account with this username
# will be created and granted administrative privileges.
# Don't use blank spaces in the username.
USER="admin_marcram"

# Please provide a password for that new administrator account
PASSWORD="mrg"

# By default this downloads 'latest' ejabberd version,
# but you can set a specific version, for example '22.05'
# or the bleeding edge 'master'. See available tags in
# https://github.com/processone/ejabberd/pkgs/container/ejabberd
VERSION="latest"

# This tells docker what ports ejabberd will use.
# You can later configure them in conf/ejabberd.yml
PORTS="5280 5222 5269 5443"

#
# 3. Now save this script and run it with: chmod +x ejabberd-installer.sh && ./ejabberd-installer.sh
#

#
# 4. When installation is completed:
#
# You can manage the ejabberd container using docker commands:
#
# - Start the ejabberd container: docker start ejabberd
# - Stop the ejabberd container: docker stop ejabberd
# - View logs: docker logs ejabberd
# - Access ejabberdctl: docker exec -it ejabberd ejabberdctl
# - Enter WebAdmin: http://localhost:5180
#
# You can delete the container and create it again by running this script,
# the configuration and database are maintained.
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}=== $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}=== $1${NC}"
}

print_error() {
    echo -e "${RED}=== ERROR: $1${NC}"
}

#===============================================================
# Check if docker is installed
#===============================================================

print_info "Checking if Docker is installed..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed!"
    echo
    echo "Please install Docker first:"
    echo "  For macOS: Download Docker Desktop from https://www.docker.com/"
    echo "  For Linux: Install using your package manager"
    echo "    Ubuntu/Debian: sudo apt install docker.io"
    echo "    CentOS/RHEL: sudo yum install docker"
    echo
    echo "Then try running this script again."
    exit 1
fi

# Check if docker daemon is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running!"
    echo
    echo "Please start Docker first:"
    echo "  For macOS: Start Docker Desktop"
    echo "  For Linux: sudo systemctl start docker"
    echo
    echo "Then try running this script again."
    exit 1
fi

#===============================================================
# Check install options are correctly set
#===============================================================

if [ -z "$PASSWORD" ]; then
    print_error "PASSWORD not set!"
    echo
    echo "Please edit this script and set the PASSWORD variable."
    echo "Then try running this script again."
    exit 1
fi

#===============================================================
# Download Docker image
#===============================================================

IMAGE="ghcr.io/processone/ejabberd:$VERSION"

print_info "Checking if the '$IMAGE' container image was already downloaded..."
if docker image inspect "$IMAGE" &> /dev/null; then
    print_info "The '$IMAGE' container image was already downloaded."
else
    print_info "The '$IMAGE' container image was not downloaded yet."
    echo
    print_info "Downloading the '$IMAGE' container image, please wait..."
    docker pull "$IMAGE"
fi

#===============================================================
# Create preliminary container
#===============================================================

print_info "Checking if the 'ejabberd' container already exists..."
if docker container inspect ejabberd &> /dev/null; then
    print_info "The 'ejabberd' container already exists."
    print_info "Nothing to do, so installation finishes now."
    print_info "You can start the 'ejabberd' container with: docker start ejabberd"
    exit 0
else
    print_info "The 'ejabberd' container doesn't yet exist,"
    print_info "so let's continue the installation process."
fi

if [ -d "$INSTALL_DIR" ]; then
    print_info "The INSTALL_DIR $INSTALL_DIR already exists."
    print_info "No need to create the preliminary 'ejabberd-pre' container."
else
    print_info "The INSTALL_DIR $INSTALL_DIR doesn't exist."
    print_info "Let's create the preliminary 'ejabberd-pre' container."
    
    # Create preliminary container function
    create_ejabberd_pre() {
        print_info "Creating a preliminary 'ejabberd-pre' container using $IMAGE image..."
        docker create --name ejabberd-pre --hostname localhost "$IMAGE"
        
        print_info "Starting 'ejabberd-pre' container..."
        docker container start ejabberd-pre
        
        print_info "Waiting for ejabberd to be running..."
        timeout=10
        status=4
        
        while [ $status -gt 0 ] && [ $timeout -gt 0 ]; do
            echo -n "."
            sleep 1
            timeout=$((timeout - 1))
            docker exec ejabberd-pre ejabberdctl status &> /dev/null
            status=$?
        done
        echo
        
        if [ $status -ne 0 ]; then
            print_error "Timeout waiting for ejabberd to start"
            return 1
        fi
        
        print_info "Setting configuration options..."
        docker exec ejabberd-pre sed -i "s/- localhost/- $HOST/g" conf/ejabberd.yml
        docker exec ejabberd-pre sed -i "s/^acl:/acl:\n  admin:\n    user:\n      - \"$USER@$HOST\"/g" conf/ejabberd.yml
        docker exec ejabberd-pre sed -i "s/5280/5180/g" conf/ejabberd.yml
        docker exec ejabberd-pre sed -i "s!/admin!/!g" conf/ejabberd.yml
        docker exec ejabberd-pre ejabberdctl reload_config
        
        print_info "Registering the administrator account..."
        docker exec ejabberd-pre ejabberdctl register "$USER" "$HOST" "$PASSWORD"
        docker exec ejabberd-pre ejabberdctl stop
        
        print_info "Copying conf, database, logs..."
        mkdir -p "$INSTALL_DIR"/{conf,database,logs,ejabberd-modules}
        docker cp ejabberd-pre:/opt/ejabberd/conf/ "$INSTALL_DIR/"
        docker cp ejabberd-pre:/opt/ejabberd/database/ "$INSTALL_DIR/"
        docker cp ejabberd-pre:/opt/ejabberd/logs/ "$INSTALL_DIR/"
        
        print_info "Deleting the preliminary 'ejabberd-pre' container..."
        docker stop ejabberd-pre 2>/dev/null || true
        docker rm ejabberd-pre
        
        return 0
    }
    
    create_ejabberd_pre
fi

#===============================================================
# Create final container
#===============================================================

print_info "Creating the final 'ejabberd' container using $IMAGE image..."

# Build port arguments
PORT_ARGS=""
for port in $PORTS; do
    PORT_ARGS="$PORT_ARGS -p $port:$port"
done

# Volume arguments
VOLUME_ARGS="-v $INSTALL_DIR/conf:/opt/ejabberd/conf"
VOLUME_ARGS="$VOLUME_ARGS -v $INSTALL_DIR/database:/opt/ejabberd/database"
VOLUME_ARGS="$VOLUME_ARGS -v $INSTALL_DIR/logs:/opt/ejabberd/logs"
VOLUME_ARGS="$VOLUME_ARGS -v $INSTALL_DIR/ejabberd-modules:/opt/ejabberd/.ejabberd-modules"

docker create --name ejabberd --hostname localhost $PORT_ARGS $VOLUME_ARGS "$IMAGE"

print_info "Installation completed successfully!"
echo
print_info "You can now manage your ejabberd container:"
echo "  Start:    docker start ejabberd"
echo "  Stop:     docker stop ejabberd"
echo "  Logs:     docker logs ejabberd"
echo "  Shell:    docker exec -it ejabberd /bin/bash"
echo "  ejabberdctl: docker exec -it ejabberd ejabberdctl"
echo "  WebAdmin: http://localhost:5180"
echo
print_info "Configuration files are stored in: $INSTALL_DIR"