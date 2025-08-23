#!/bin/bash
set -euo pipefail


SCRIPT_VERSION="1.0.0"
INSTALL_DIR="${INSTALL_DIR:-$HOME/alphaquanta_q}"
LOG_FILE="/tmp/alphaquanta_install.log"
CI_MODE="${CI_MODE:-false}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

check_prerequisites() {
    log "Checking system prerequisites..."
    
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        error "This installer requires Linux. Detected: $OSTYPE"
    fi
    
    ARCH=$(uname -m)
    if [[ "$ARCH" != "x86_64" ]]; then
        warn "Untested architecture: $ARCH. Proceeding anyway."
    fi
    
    local required_commands=("curl" "git" "docker" "python3")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error "Required command not found: $cmd"
        fi
    done
    
    if ! docker info &> /dev/null; then
        error "Docker daemon not running. Please start Docker and try again."
    fi
    
    success "Prerequisites check passed"
}

check_quantum_tokens() {
    log "Checking quantum computing credentials..."
    
    local tokens_found=0
    
    if [[ -n "${IBM_QUANTUM_TOKEN:-}" ]]; then
        log "✅ IBM Quantum token detected"
        tokens_found=$((tokens_found + 1))
    else
        warn "IBM_QUANTUM_TOKEN not set. Quantum features will be limited."
    fi
    
    if [[ -n "${DWAVE_API_TOKEN:-}" ]]; then
        log "✅ D-Wave API token detected"
        tokens_found=$((tokens_found + 1))
    else
        warn "DWAVE_API_TOKEN not set. D-Wave features will be disabled."
    fi
    
    if [[ $tokens_found -eq 0 ]]; then
        warn "No quantum tokens found. Running in classical-only mode."
        export QUANTUM_ENABLED=false
    else
        log "Quantum computing enabled with $tokens_found provider(s)"
        export QUANTUM_ENABLED=true
    fi
}

install_python_dependencies() {
    log "Installing Python dependencies..."
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$PYTHON_VERSION < 3.12" | bc -l) -eq 1 ]]; then
        error "Python 3.12+ required. Found: $PYTHON_VERSION"
    fi
    
    if ! command -v poetry &> /dev/null; then
        log "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    cd "$INSTALL_DIR"
    poetry install --no-dev
    
    success "Python dependencies installed"
}

setup_environment() {
    log "Setting up environment configuration..."
    
    if [[ ! -f "$INSTALL_DIR/.env" ]]; then
        cat > "$INSTALL_DIR/.env" << EOF

IBM_QUANTUM_TOKEN=${IBM_QUANTUM_TOKEN:-}
DWAVE_API_TOKEN=${DWAVE_API_TOKEN:-}

QC_API_TOKEN=${QC_API_TOKEN:-}

IB_USERNAME=${IB_USERNAME:-}
IB_PASSWORD=${IB_PASSWORD:-}

GRAFANA_PASSWORD=${GRAFANA_PASSWORD:-alphaquanta}
N8N_USER=${N8N_USER:-admin}
N8N_PASSWORD=${N8N_PASSWORD:-alphaquanta}
VNC_PASSWORD=${VNC_PASSWORD:-alphaquanta}

QUANTUM_ENABLED=${QUANTUM_ENABLED:-true}
TRADING_MODE=${TRADING_MODE:-paper}
LOG_LEVEL=${LOG_LEVEL:-INFO}
EOF
        log "Created .env file. Please update with your credentials."
    else
        log "Using existing .env file"
    fi
    
    mkdir -p "$INSTALL_DIR"/{data,algorithms,results,quantum,monitoring,workflows,scripts}
    
    success "Environment setup complete"
}

setup_monitoring() {
    log "Setting up monitoring configuration..."
    
    mkdir -p "$INSTALL_DIR/monitoring"
    cat > "$INSTALL_DIR/monitoring/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'alphaquanta-lean'
    static_configs:
      - targets: ['lean-engine:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'alphaquanta-qpu'
    static_configs:
      - targets: ['qiskit-runtime:8000']
    metrics_path: '/qpu/metrics'
    scrape_interval: 60s
EOF

    success "Monitoring configuration created"
}

setup_n8n_workflow() {
    log "Setting up n8n workflow automation..."
    
    mkdir -p "$INSTALL_DIR/workflows"
    
    cat > "$INSTALL_DIR/workflows/itip1_workflow.json" << EOF
{
  "name": "AlphaQuanta Trading Pipeline",
  "nodes": [
    {
      "parameters": {},
      "name": "Start",
      "type": "n8n-nodes-base.start",
      "typeVersion": 1,
      "position": [240, 300]
    },
    {
      "parameters": {
        "httpMethod": "POST",
        "path": "/webhook/trading-signal",
        "responseMode": "responseNode"
      },
      "name": "Trading Signal Webhook",
      "type": "n8n-nodes-base.webhook",
      "typeVersion": 1,
      "position": [460, 300]
    }
  ],
  "connections": {},
  "active": true,
  "settings": {},
  "id": "alphaquanta-trading-pipeline"
}
EOF
    
    success "n8n workflow configuration created"
}

start_services() {
    log "Starting AlphaQuanta services..."
    
    cd "$INSTALL_DIR"
    
    docker-compose pull
    
    if [[ "$CI_MODE" == "true" ]]; then
        log "Starting services in CI mode..."
        docker-compose up -d --scale qiskit-runtime=0
    else
        docker-compose up -d
    fi
    
    log "Waiting for services to become healthy..."
    local max_wait=300  # 5 minutes
    local wait_time=0
    
    while [[ $wait_time -lt $max_wait ]]; do
        local healthy_count=$(docker-compose ps --services --filter "status=running" | wc -l)
        local total_services=$(docker-compose config --services | wc -l)
        
        if [[ "$CI_MODE" == "true" ]]; then
            total_services=$((total_services - 1))  # Exclude qiskit-runtime in CI
        fi
        
        if [[ $healthy_count -eq $total_services ]]; then
            success "All services are healthy"
            break
        fi
        
        log "Waiting for services... ($healthy_count/$total_services healthy)"
        sleep 10
        wait_time=$((wait_time + 10))
    done
    
    if [[ $wait_time -ge $max_wait ]]; then
        error "Services failed to start within $max_wait seconds"
    fi
}

run_health_checks() {
    log "Running health checks..."
    
    local checks_passed=0
    local total_checks=4
    
    if curl -sf http://localhost:8080/healthz &> /dev/null; then
        log "✅ Lean engine healthy"
        checks_passed=$((checks_passed + 1))
    else
        warn "❌ Lean engine not responding"
    fi
    
    if curl -sf http://localhost:9090/-/healthy &> /dev/null; then
        log "✅ Prometheus healthy"
        checks_passed=$((checks_passed + 1))
    else
        warn "❌ Prometheus not responding"
    fi
    
    if curl -sf http://localhost:3000/api/health &> /dev/null; then
        log "✅ Grafana healthy"
        checks_passed=$((checks_passed + 1))
    else
        warn "❌ Grafana not responding"
    fi
    
    if curl -sf http://localhost:5678/healthz &> /dev/null; then
        log "✅ n8n healthy"
        checks_passed=$((checks_passed + 1))
    else
        warn "❌ n8n not responding"
    fi
    
    if [[ $checks_passed -eq $total_checks ]]; then
        success "All health checks passed ($checks_passed/$total_checks)"
    else
        warn "Some health checks failed ($checks_passed/$total_checks)"
    fi
}

post_install_workflow() {
    log "Importing n8n workflow..."
    
    sleep 30
    
    if curl -sf -X POST \
        -H "Content-Type: application/json" \
        -d @"$INSTALL_DIR/workflows/itip1_workflow.json" \
        http://localhost:5678/api/v1/workflows &> /dev/null; then
        success "n8n workflow imported successfully"
    else
        warn "Failed to import n8n workflow automatically"
    fi
}

cleanup_on_error() {
    if [[ $? -ne 0 ]]; then
        error "Installation failed. Check logs at $LOG_FILE"
        log "Cleaning up..."
        cd "$INSTALL_DIR" && docker-compose down &> /dev/null || true
    fi
}

main() {
    trap cleanup_on_error ERR
    
    log "🚀 AlphaQuanta Quantum-Hybrid Trading Stack Installer v$SCRIPT_VERSION"
    log "Installation directory: $INSTALL_DIR"
    log "CI Mode: $CI_MODE"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --ci)
                CI_MODE=true
                shift
                ;;
            --install-dir)
                INSTALL_DIR="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [--ci] [--install-dir DIR] [--help]"
                echo "  --ci           Run in CI mode (mock quantum services)"
                echo "  --install-dir  Installation directory (default: $HOME/alphaquanta_q)"
                echo "  --help         Show this help message"
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
    
    if [[ ! -d "$INSTALL_DIR" ]]; then
        log "Cloning AlphaQuanta repository..."
        git clone https://github.com/QuantConnect/alphaquanta_q.git "$INSTALL_DIR"
    else
        log "Using existing installation directory"
        cd "$INSTALL_DIR"
        git pull origin main &> /dev/null || log "Could not update repository"
    fi
    
    check_prerequisites
    check_quantum_tokens
    setup_environment
    install_python_dependencies
    setup_monitoring
    setup_n8n_workflow
    start_services
    run_health_checks
    
    if [[ "$CI_MODE" != "true" ]]; then
        post_install_workflow
    fi
    
    success "🎉 AlphaQuanta installation complete!"
    log ""
    log "📊 Access Points:"
    log "   • Grafana Dashboard: http://localhost:3000 (admin/alphaquanta)"
    log "   • Prometheus Metrics: http://localhost:9090"
    log "   • n8n Workflows: http://localhost:5678 (admin/alphaquanta)"
    log "   • IB Gateway VNC: vnc://localhost:5900 (alphaquanta)"
    log ""
    log "🚀 Quick Start:"
    log "   cd $INSTALL_DIR"
    log "   python runner.py --mode paper --symbol SPY"
    log ""
    log "📝 Configuration:"
    log "   Edit $INSTALL_DIR/.env to set your API tokens"
    log "   Edit $INSTALL_DIR/qconfig.yaml for quantum settings"
    log ""
    log "📋 Logs: $LOG_FILE"
    
    INSTALL_CHECKSUM=$(sha256sum "$0" 2>/dev/null | cut -d' ' -f1 || echo "unknown")
    log "📋 Install script checksum: $INSTALL_CHECKSUM"
}

main "$@"
