#!/bin/bash
# =============================================================================
# Qwen Fine Tune Test Suite - Launcher
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║           Qwen Fine Tune Test Suite                        ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_help() {
    echo "Usage: ./launch.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  dev         Start development servers (backend + frontend)"
    echo "  backend     Start backend API only (FastAPI on :8000)"
    echo "  frontend    Start frontend only (Vite on :5173)"
    echo "  docker      Start with Docker Compose (production)"
    echo "  docker-dev  Start with Docker (development mode)"
    echo "  stop        Stop all running services"
    echo "  install     Install all dependencies"
    echo "  streamlit   Start legacy Streamlit app"
    echo "  train       Run interactive training"
    echo "  evaluate    Run model evaluation"
    echo "  convert     Run GGUF conversion"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./launch.sh dev        # Development with hot reload"
    echo "  ./launch.sh docker     # Production with Docker"
    echo ""
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python 3 is not installed${NC}"
        exit 1
    fi
}

check_node() {
    if ! command -v node &> /dev/null; then
        echo -e "${YELLOW}Warning: Node.js is not installed. Frontend won't work.${NC}"
        return 1
    fi
    return 0
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi
}

install_deps() {
    echo -e "${CYAN}Installing dependencies...${NC}"

    # Backend
    echo -e "${GREEN}Installing backend dependencies...${NC}"
    pip install -r backend/requirements.txt

    # Frontend
    if check_node; then
        echo -e "${GREEN}Installing frontend dependencies...${NC}"
        cd frontend && npm install && cd ..
    fi

    echo -e "${GREEN}Dependencies installed!${NC}"
}

start_backend() {
    echo -e "${GREEN}Starting backend on http://localhost:8000${NC}"
    echo -e "${CYAN}API docs: http://localhost:8000/docs${NC}"
    cd backend
    python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
}

start_frontend() {
    if ! check_node; then
        echo -e "${RED}Cannot start frontend without Node.js${NC}"
        exit 1
    fi
    echo -e "${GREEN}Starting frontend on http://localhost:5173${NC}"
    cd frontend && npm run dev
}

start_dev() {
    echo -e "${GREEN}Starting development servers...${NC}"
    echo -e "${CYAN}Backend:  http://localhost:8000${NC}"
    echo -e "${CYAN}Frontend: http://localhost:5173${NC}"
    echo -e "${CYAN}API Docs: http://localhost:8000/docs${NC}"
    echo ""

    # Start backend in background
    cd backend
    python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    cd ..

    # Give backend time to start
    sleep 2

    # Start frontend
    if check_node; then
        cd frontend && npm run dev &
        FRONTEND_PID=$!
        cd ..
    else
        echo -e "${YELLOW}Skipping frontend (Node.js not installed)${NC}"
    fi

    # Trap to cleanup on exit
    trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT

    echo ""
    echo -e "${GREEN}Services started! Press Ctrl+C to stop.${NC}"

    # Wait for processes
    wait
}

start_docker() {
    check_docker
    echo -e "${GREEN}Starting with Docker Compose...${NC}"
    echo -e "${CYAN}Frontend: http://localhost:3000${NC}"
    echo -e "${CYAN}Backend:  http://localhost:8000${NC}"
    docker compose up --build frontend backend
}

start_docker_dev() {
    check_docker
    echo -e "${GREEN}Starting Docker in development mode...${NC}"
    docker compose up --build
}

stop_services() {
    echo -e "${YELLOW}Stopping services...${NC}"

    # Stop Docker containers
    docker compose down 2>/dev/null || true

    # Kill uvicorn processes
    pkill -f "uvicorn app.main:app" 2>/dev/null || true

    # Kill vite processes
    pkill -f "vite" 2>/dev/null || true

    # Kill node processes in frontend
    pkill -f "node.*frontend" 2>/dev/null || true

    echo -e "${GREEN}Services stopped.${NC}"
}

start_streamlit() {
    check_python
    echo -e "${GREEN}Starting Streamlit on http://localhost:8501${NC}"
    streamlit run app.py
}

run_train() {
    check_python
    echo -e "${GREEN}Starting interactive training...${NC}"
    python train.py
}

run_evaluate() {
    check_python
    echo -e "${GREEN}Starting model evaluation...${NC}"
    python evaluate_model.py
}

run_convert() {
    check_python
    echo -e "${GREEN}Starting GGUF conversion...${NC}"
    python merge_and_convert_gguff.py
}

# Main
print_banner

case "${1:-help}" in
    dev)
        check_python
        start_dev
        ;;
    backend)
        check_python
        start_backend
        ;;
    frontend)
        start_frontend
        ;;
    docker)
        start_docker
        ;;
    docker-dev)
        start_docker_dev
        ;;
    stop)
        stop_services
        ;;
    install)
        install_deps
        ;;
    streamlit)
        start_streamlit
        ;;
    train)
        run_train
        ;;
    evaluate)
        run_evaluate
        ;;
    convert)
        run_convert
        ;;
    help|--help|-h|"")
        print_help
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        echo ""
        print_help
        exit 1
        ;;
esac
