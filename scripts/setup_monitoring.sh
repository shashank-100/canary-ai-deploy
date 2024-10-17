#!/bin/bash

# Monitoring Stack Setup Script for AI Model Serving Platform

set -e

# Configuration
NAMESPACE="monitoring"
PROMETHEUS_CHART_VERSION="51.3.0"
LOKI_CHART_VERSION="5.36.0"
GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-admin123}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log_info "Creating monitoring namespace..."
    
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Namespace $NAMESPACE created/updated"
}

# Add Helm repositories
add_helm_repos() {
    log_info "Adding Helm repositories..."
    
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    log_success "Helm repositories added and updated"
}

# Install Prometheus stack
install_prometheus() {
    log_info "Installing Prometheus stack..."
    
    # Check if already installed
    if helm list -n $NAMESPACE | grep -q "prometheus"; then
        log_warning "Prometheus stack already installed, upgrading..."
        helm upgrade prometheus prometheus-community/kube-prometheus-stack \
            --namespace $NAMESPACE \
            --values k8s/addons/monitoring/prometheus-values.yaml \
            --version $PROMETHEUS_CHART_VERSION \
            --wait \
            --timeout 10m
    else
        helm install prometheus prometheus-community/kube-prometheus-stack \
            --namespace $NAMESPACE \
            --values k8s/addons/monitoring/prometheus-values.yaml \
            --version $PROMETHEUS_CHART_VERSION \
            --wait \
            --timeout 10m
    fi
    
    log_success "Prometheus stack installed/upgraded"
}

# Install Loki stack
install_loki() {
    log_info "Installing Loki stack..."
    
    # Check if already installed
    if helm list -n $NAMESPACE | grep -q "loki"; then
        log_warning "Loki stack already installed, upgrading..."
        helm upgrade loki grafana/loki-stack \
            --namespace $NAMESPACE \
            --values k8s/addons/monitoring/loki-values.yaml \
            --version $LOKI_CHART_VERSION \
            --wait \
            --timeout 10m
    else
        helm install loki grafana/loki-stack \
            --namespace $NAMESPACE \
            --values k8s/addons/monitoring/loki-values.yaml \
            --version $LOKI_CHART_VERSION \
            --wait \
            --timeout 10m
    fi
    
    log_success "Loki stack installed/upgraded"
}

# Apply monitoring configurations
apply_monitoring_configs() {
    log_info "Applying monitoring configurations..."
    
    # Apply Grafana dashboards
    kubectl apply -f k8s/addons/monitoring/grafana-dashboards.yaml
    
    # Apply Prometheus rules
    kubectl apply -f k8s/addons/monitoring/prometheus-rules.yaml
    
    # Apply ServiceMonitors
    kubectl apply -f k8s/addons/monitoring/servicemonitor.yaml
    
    log_success "Monitoring configurations applied"
}

# Wait for deployments
wait_for_deployments() {
    log_info "Waiting for deployments to be ready..."
    
    # Wait for Prometheus
    kubectl wait --for=condition=available --timeout=300s deployment/prometheus-kube-prometheus-operator -n $NAMESPACE
    kubectl wait --for=condition=available --timeout=300s deployment/prometheus-grafana -n $NAMESPACE
    
    # Wait for Loki
    kubectl wait --for=condition=available --timeout=300s deployment/loki -n $NAMESPACE
    kubectl wait --for=condition=ready --timeout=300s pod -l app=promtail -n $NAMESPACE
    
    log_success "All deployments are ready"
}

# Setup port forwarding
setup_port_forwarding() {
    log_info "Setting up port forwarding..."
    
    # Kill existing port forwards
    pkill -f "kubectl port-forward" || true
    sleep 2
    
    # Grafana port forward
    kubectl port-forward -n $NAMESPACE svc/prometheus-grafana 3000:80 &
    GRAFANA_PID=$!
    
    # Prometheus port forward
    kubectl port-forward -n $NAMESPACE svc/prometheus-kube-prometheus-prometheus 9090:9090 &
    PROMETHEUS_PID=$!
    
    # Alertmanager port forward
    kubectl port-forward -n $NAMESPACE svc/prometheus-kube-prometheus-alertmanager 9093:9093 &
    ALERTMANAGER_PID=$!
    
    # Save PIDs for cleanup
    echo $GRAFANA_PID > /tmp/grafana.pid
    echo $PROMETHEUS_PID > /tmp/prometheus.pid
    echo $ALERTMANAGER_PID > /tmp/alertmanager.pid
    
    log_success "Port forwarding setup complete"
}

# Get access information
get_access_info() {
    log_info "Getting access information..."
    
    echo ""
    echo "=== Monitoring Stack Access Information ==="
    echo ""
    
    # Grafana
    echo "Grafana:"
    echo "  URL: http://localhost:3000"
    echo "  Username: admin"
    echo "  Password: $GRAFANA_ADMIN_PASSWORD"
    echo ""
    
    # Prometheus
    echo "Prometheus:"
    echo "  URL: http://localhost:9090"
    echo ""
    
    # Alertmanager
    echo "Alertmanager:"
    echo "  URL: http://localhost:9093"
    echo ""
    
    # Kubernetes services
    echo "Kubernetes Services:"
    kubectl get svc -n $NAMESPACE
    echo ""
    
    # Pods status
    echo "Pods Status:"
    kubectl get pods -n $NAMESPACE
    echo ""
    
    log_success "Access information displayed"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Check Prometheus targets
    log_info "Checking Prometheus targets..."
    sleep 10  # Wait for services to be discovered
    
    # Check if Grafana is accessible
    if curl -s http://localhost:3000/api/health > /dev/null; then
        log_success "Grafana is accessible"
    else
        log_warning "Grafana may not be ready yet"
    fi
    
    # Check if Prometheus is accessible
    if curl -s http://localhost:9090/-/healthy > /dev/null; then
        log_success "Prometheus is accessible"
    else
        log_warning "Prometheus may not be ready yet"
    fi
    
    log_success "Installation verification complete"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up port forwards..."
    
    if [ -f /tmp/grafana.pid ]; then
        kill $(cat /tmp/grafana.pid) 2>/dev/null || true
        rm /tmp/grafana.pid
    fi
    
    if [ -f /tmp/prometheus.pid ]; then
        kill $(cat /tmp/prometheus.pid) 2>/dev/null || true
        rm /tmp/prometheus.pid
    fi
    
    if [ -f /tmp/alertmanager.pid ]; then
        kill $(cat /tmp/alertmanager.pid) 2>/dev/null || true
        rm /tmp/alertmanager.pid
    fi
}

# Trap cleanup on exit
trap cleanup EXIT

# Main execution
main() {
    log_info "Starting monitoring stack setup..."
    
    check_prerequisites
    create_namespace
    add_helm_repos
    install_prometheus
    install_loki
    apply_monitoring_configs
    wait_for_deployments
    setup_port_forwarding
    get_access_info
    verify_installation
    
    log_success "Monitoring stack setup completed successfully!"
    log_info "Press Ctrl+C to stop port forwarding and exit"
    
    # Keep script running for port forwarding
    wait
}

# Handle command line arguments
case "${1:-}" in
    "install")
        main
        ;;
    "uninstall")
        log_info "Uninstalling monitoring stack..."
        helm uninstall prometheus -n $NAMESPACE || true
        helm uninstall loki -n $NAMESPACE || true
        kubectl delete namespace $NAMESPACE || true
        log_success "Monitoring stack uninstalled"
        ;;
    "status")
        log_info "Checking monitoring stack status..."
        kubectl get all -n $NAMESPACE
        ;;
    "port-forward")
        setup_port_forwarding
        get_access_info
        log_info "Press Ctrl+C to stop port forwarding"
        wait
        ;;
    *)
        echo "Usage: $0 {install|uninstall|status|port-forward}"
        echo ""
        echo "Commands:"
        echo "  install      - Install the complete monitoring stack"
        echo "  uninstall    - Uninstall the monitoring stack"
        echo "  status       - Show status of monitoring components"
        echo "  port-forward - Setup port forwarding for local access"
        exit 1
        ;;
esac

