# ModelServeAI with Canary Deployments

This project aims to build a robust, scalable, and observable platform for serving AI models as microservices on AWS, leveraging Kubernetes for orchestration and Terraform for infrastructure as code. The platform will support dynamic scaling, A/B testing, and real-time monitoring, enabling efficient deployment and management of various AI models (e.g., NLP, CV).

## Key Features:

- **Infrastructure as Code (Terraform):** Automated provisioning of AWS resources, including EKS, PostgreSQL, Redis, S3, VPC, and Load Balancers.
- **Kubernetes (K8s) Configuration:** Deployment and management of AI model microservices using FastAPI containers, with advanced features like HPA, KEDA, Istio for canary deployments, and Nginx Ingress for routing.
- **Python Components:** FastAPI-based model serving endpoints, custom metrics exposure, shadow mode for A/B testing, and data drift detection.
- **CI/CD (GitHub Actions):** Automated build, test, vulnerability scanning, and deployment pipelines with support for canary releases and auto-rollback.
- **Observability Stack:** Comprehensive monitoring and logging with Prometheus, Grafana, and Loki for real-time insights into model performance and platform health.
- **Advanced Features:** Support for multi-model pipelines, GPU fractionalization, and cost optimization strategies.

## Architecture Overview:

(Detailed architecture diagram will be included in HLD.md)

## Getting Started:

(Detailed setup and deployment instructions will be provided in subsequent documentation.)

## Quick Start

### Prerequisites
- AWS CLI configured with appropriate permissions
- Terraform >= 1.5
- kubectl >= 1.27
- Helm >= 3.12
- Docker >= 20.10

### 1. Infrastructure Deployment
```bash
# Clone the repository
git clone <repository-url>
cd ai-model-serving-platform

# Configure environment variables
export AWS_REGION=us-west-2
export CLUSTER_NAME=ai-model-serving
export ENVIRONMENT=dev

# Deploy infrastructure
cd terraform/environments/dev
terraform init
terraform plan
terraform apply
```

### 2. Kubernetes Setup
```bash
# Configure kubectl
aws eks update-kubeconfig --region $AWS_REGION --name $CLUSTER_NAME

# Deploy base Kubernetes resources
kubectl apply -f k8s/base/

# Deploy environment-specific configurations
kubectl apply -k k8s/overlays/dev/
```

### 3. Monitoring Stack
```bash
# Install monitoring components
./scripts/setup_monitoring.sh install

# Access Grafana (after port-forward)
# URL: http://localhost:3000

> **Note:**  
> The Grafana username and password are set during deployment.  
> - For local/dev, check your deployment values or environment variables.  
> - For production, credentials should be securely managed (e.g., via a secrets manager or environment variables) and never shared publicly.
>
> **Default credentials (if not changed):**
> - Username: `admin`
> - Password: (see your deployment configuration)
>
> **Important:**  
> Always change default credentials before deploying to production.

The Grafana admin password is set in your Helm values file (`adminPassword`) or via a Kubernetes secret.
```
```

### 4. Model Deployment
```bash
# Build and push model images
docker build -t $ECR_REGISTRY/bert-ner:latest docker/bert_ner/
docker push $ECR_REGISTRY/bert-ner:latest

# Deploy models
kubectl apply -f k8s/base/deployment.yaml
```

## Architecture Deep Dive

### Infrastructure Layer

The platform is built on a robust AWS infrastructure foundation that provides scalability, reliability, and security. The infrastructure uses Amazon EKS as the container orchestration platform, providing managed Kubernetes control plane with automatic updates and high availability.

Core Infrastructure

Networking: Multi-AZ VPC with public (load balancers/NAT) and private (apps/DB) subnets

Compute: Mixed node groups (general, memory-optimized, GPU, spot) for workload diversity

Storage: EBS for block storage, S3 for models/data with lifecycle policies

Managed Services: RDS PostgreSQL, ElastiCache Redis, CloudWatch monitoring

Application Layer

Microservices: Containerized components with K8s scaling

Model Serving: GPU-accelerated APIs (BERT NER, ResNet)

Orchestration: Redis-backed pipeline workflows (sync/async)

API Gateway: ALB-integrated with auth/rate limiting

Traffic Management

Istio Mesh: Sidecar proxies for security/observability

Deployments: Canary releases with traffic splitting

Resilience: Circuit breakers, least-conn LB

Observability

Metrics: Prometheus (app/infra KPIs)

Logs: Loki/Promtail with structured logging

Traces: Jaeger for distributed tracing

Alerts: Prometheus rules → AlertManager

Advanced Capabilities

Pipelines: Argo Workflows with error recovery

GPU Optimization: Sharing, dynamic allocation

Cost Control: Spot instances, right-sizing, scheduled scaling

Security

Network: VPC isolation, mTLS, WAF

IAM: K8s RBAC + AWS IRSA

Data: KMS encryption (at rest/transit), secret rotation

Performance

Inference: Model caching, dynamic batching

Infra: NUMA-aware scheduling, SR-IOV networking

## Table of Contents
- [Common Issues & Fixes](#-common-issues--fixes)
  - [Pod Startup Failures](#-pod-startup-failures)
  - [Service Connection Issues](#-service-connection-issues)
  - [Performance Problems](#-performance-problems)
- [Diagnostic Toolkit](#-diagnostic-toolkit)
  - [Basic Checks](#-basic-checks)
  - [Monitoring Tools](#-monitoring-tools)
  - [Security Checks](#-security-checks)
- [Pro Tips](#-pro-tips)

## 🚀 Common Issues & Fixes

### 🐛 Pod Startup Failures

**Symptoms:**
- Pod stuck in "Pending" or "CrashLoopBackOff" state
- Image pull errors
- Resource allocation errors

#### Diagnostic commands:
```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name> --tail=50
kubectl get events --sort-by=.metadata.creationTimestamp
```

| **Issue**               | **Fix**                                                           |
|------------------------|--------------------------------------------------------------------|
| Image pull errors       | Verify registry credentials in your secret                        |
| Insufficient resources  | Check node capacity with `kubectl describe nodes`                 |
| Config errors           | Validate environment variables and config maps        


### 🔍 Symptoms
- "Connection refused" errors  
- Intermittent timeouts  
- DNS resolution failures

## Test connectivity:

```
kubectl run network-test --image=alpine/curl --rm -it -- curl http://<service-name>.<namespace>.svc.cluster.local
```

## Performance problems :

- Symptoms:

- High latency

- CPU throttling

- OOM kills
```
kubectl top pods
kubectl describe node | grep -A 10 "Allocated resources"
```

- Solutions :

- Scale horizontally: kubectl scale deployment <name> --replicas=3

- Adjust resources in deployment YAML

- Enable HPA: kubectl autoscale deployment <name> --cpu-percent=50 --min=2 --max=5

Troubleshooting Steps:

- Verify service exists: kubectl get svc

- Check endpoints: kubectl get endpoints service-name

- Test DNS resolution inside cluster

### Monitoring Tools

#### Access Prometheus
```
kubectl port-forward svc/prometheus 9090
```

#### View logs
```
kubectl logs -f <pod-name> --container <container-name>
```
### Security Checks

#### Verify permissions
```
kubectl auth can-i create deployments
```

#### Check network policies
```
kubectl get networkpolicy -A
```

## Pro Tips

##### Get detailed pod info:
```
kubectl get pod <pod-name> -o yaml
```

##### Stream logs from multiple pods:
```
kubectl logs -f -l app=<app-label> --all-containers
```

### Useful Aliases
```
alias k='kubectl'
alias kgp='kubectl get pods'
alias kdp='kubectl describe pod'
```

## Contributing

### Local Development Setup

1. **Clone the Repository**
   ```sh
   git clone <your-repo-url>
   cd ai-model-serving-platform
   ```

2. **Install Prerequisites**
   - [Docker Desktop](https://www.docker.com/products/docker-desktop) (includes Docker & Kubernetes)
   - [kubectl](https://kubernetes.io/docs/tasks/tools/)
   - [Kind](https://kind.sigs.k8s.io/) or [Minikube](https://minikube.sigs.k8s.io/docs/)
   - [Terraform](https://www.terraform.io/downloads)
   - [Helm](https://helm.sh/docs/intro/install/)
   - [Python 3.8+](https://www.python.org/downloads/)

3. **Start a Local Kubernetes Cluster**
   - Using Kind:
     ```sh
     kind create cluster --name modelserve
     ```
   - Or using Minikube:
     ```sh
     minikube start
     ```

4. **Build and Load Docker Images**
   - Build your model server images:
     ```sh
     docker build -t bert-ner-model:dev ./docker/bert_ner
     docker build -t resnet-classifier-model:dev ./docker/resnet_classifier
     ```
   - Load images into Kind (if using Kind):
     ```sh
     kind load docker-image bert-ner-model:dev --name modelserve
     kind load docker-image resnet-classifier-model:dev --name modelserve
     ```

5. **Deploy Kubernetes Resources**
   ```sh
   kubectl apply -k k8s/base
   kubectl apply -k k8s/overlays/dev
   ```

6. **(Optional) Deploy Monitoring Stack**
   ```sh
   kubectl apply -k k8s/addons/monitoring
   ```

7. **Access Services**
   - Port-forward to access a model API or Grafana:
     ```sh
     kubectl port-forward svc/bert-ner-model-service -n ai-models 8000:8000
     kubectl port-forward svc/grafana -n monitoring 3000:3000
     ```

8. **Run Tests or Scripts**
   - (If needed) Run Python scripts for canary or drift detection:
     ```sh
     python scripts/canary_tests.py
     ```

**Tip:**
- Check the `README.md` for environment variables or secrets you may need to set.
- For troubleshooting, use `kubectl get pods -A` and `kubectl logs <pod> -n <namespace>`.

### Contribution Guidelines

Fork the repository

- Create feature branch (git checkout -b feat/your-feature)

- Commit changes (git commit -s -m "feat: your description")

- Push to branch (git push origin feat/your-feature) 

- Ensure all changes are properly tested before submission. Unit tests cover individual functions and components. Integration tests verify component interactions. End-to-end tests validate complete workflows and user scenarios.

- Ensure all changes are properly documented. Code comments explain complex logic and design decisions. README files provide clear setup and usage instructions. API documentation describes all endpoints and parameters.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support and questions:

- **Documentation**: Comprehensive guides and API documentation
- **Issues**: GitHub Issues for bug reports and feature requests  
- **Discussions**: GitHub Discussions for questions and community support
- **Security**: security@ai-model-serving.com for security-related issues

## Acknowledgments

- **Kubernetes Community** for the robust container orchestration platform
- **Prometheus Community** for comprehensive monitoring capabilities
- **Istio Community** for advanced service mesh functionality
- **AWS** for reliable cloud infrastructure services
- **Open Source Contributors** who make projects like this possible

---

**Built with ❤️ by the [pydevsg](https://github.com/pydevsg/)**

