# High-Level Design (HLD)

## 1. Introduction

This document outlines the high-level design for the ModelServeAI. It details the architecture, technology stack, and key components of the system, providing a comprehensive overview of the platform's design and functionality.

## 2. Architecture

The platform will be built on a microservices architecture, with each AI model deployed as a separate service. This approach provides flexibility, scalability, and fault isolation. The core components of the architecture are:

- **Cloud Infrastructure (AWS):** The platform will be hosted on AWS, leveraging its managed services for scalability, reliability, and security.
- **Container Orchestration (Kubernetes):** Kubernetes (EKS) will be used to orchestrate the deployment, scaling, and management of the model serving containers.
- **Service Mesh (Istio):** Istio will be used to manage traffic between the microservices, enabling features like canary deployments, A/B testing, and traffic shifting.
- **Observability Stack:** A combination of Prometheus, Grafana, and Loki will be used to provide a comprehensive monitoring and logging solution.

(A detailed architecture diagram will be added here later)

## 3. Technology Stack

- **Infrastructure as Code:** Terraform
- **Cloud Provider:** AWS (EKS, S3, RDS for PostgreSQL, ElastiCache for Redis, CloudWatch)
- **Container Orchestration:** Kubernetes
- **Service Mesh:** Istio
- **Ingress Controller:** Nginx
- **Model Serving:** Python, FastAPI, PyTorch/TensorFlow
- **CI/CD:** GitHub Actions
- **Monitoring:** Prometheus, Grafana
- **Logging:** Loki
- **Vulnerability Scanning:** Trivy

## 4. System Components

### 4.1. Infrastructure (Terraform)

- **EKS Cluster:** An EKS cluster with auto-scaling node groups (both CPU and GPU based) will be provisioned to run the Kubernetes workloads.
- **Managed Services:** AWS managed services like RDS for PostgreSQL (for model metadata), ElastiCache for Redis (for caching predictions), and S3 (for storing model artifacts) will be used.
- **Networking:** A custom VPC with public and private subnets will be created. An Application Load Balancer (ALB) will be used to expose the services to the internet.

### 4.2. Kubernetes Workloads

- **Model Servers:** Each AI model will be packaged as a Docker container and deployed as a Kubernetes Deployment. The model serving application will be built using FastAPI.
- **Autoscaling:** Horizontal Pod Autoscaler (HPA) will be used to scale the model deployments based on CPU/memory usage. KEDA will be used to scale down the deployments to zero when they are not in use.
- **Traffic Management:** Istio's VirtualServices and DestinationRules will be used to implement canary deployments and A/B testing. Nginx Ingress will be used to route external traffic to the appropriate services.

### 4.3. Python Components

- **FastAPI Application:** The FastAPI application will expose `/predict` and `/batch-predict` endpoints for synchronous and asynchronous predictions. It will also expose custom Prometheus metrics.
- **Shadow Mode:** A shadow mode feature will be implemented to route a small percentage of traffic to a new model version for testing without impacting users.
- **Drift Detection:** A Kubernetes CronJob will run a Python script periodically to detect data drift.

### 4.4. CI/CD Pipeline

- **CI:** The CI pipeline will be triggered on every push to the repository. It will lint the code, run unit tests, build Docker images, and scan them for vulnerabilities.
- **CD:** The CD pipeline will be triggered on every merge to the `main` branch. It will deploy the new model version as a canary release, run integration tests, and gradually shift traffic to the new version. It will also have an auto-rollback mechanism in case of high error rates.

### 4.5. Observability

- **Prometheus:** Prometheus will be used to scrape metrics from the model servers, Istio, and other components.
- **Grafana:** Grafana will be used to visualize the metrics collected by Prometheus. Dashboards will be created to monitor key metrics like latency, throughput, and error rates.
- **Loki:** Loki will be used to centralize the logs from all the pods in the cluster.


