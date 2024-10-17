#!/bin/bash

# User Data Script for EKS Spot Instances
# Configures spot instances to join the EKS cluster

set -o xtrace

# Variables from Terraform
CLUSTER_NAME="${cluster_name}"
NODE_GROUP_NAME="${node_group_name}"
KUBERNETES_VERSION="${kubernetes_version}"
REGION="${region}"

# Update system
yum update -y

# Install required packages
yum install -y \
    awscli \
    jq \
    wget \
    curl \
    unzip \
    htop \
    iotop \
    docker

# Start and enable Docker
systemctl start docker
systemctl enable docker

# Add ec2-user to docker group
usermod -a -G docker ec2-user

# Install kubectl
curl -o kubectl https://amazon-eks.s3.us-west-2.amazonaws.com/1.27.1/2023-04-19/bin/linux/amd64/kubectl
chmod +x ./kubectl
mv ./kubectl /usr/local/bin/

# Install AWS IAM Authenticator
curl -o aws-iam-authenticator https://amazon-eks.s3.us-west-2.amazonaws.com/1.27.1/2023-04-19/bin/linux/amd64/aws-iam-authenticator
chmod +x ./aws-iam-authenticator
mv ./aws-iam-authenticator /usr/local/bin/

# Configure kubelet
mkdir -p /etc/kubernetes/kubelet
mkdir -p /var/lib/kubelet
mkdir -p /var/lib/kubernetes

# Get cluster endpoint and certificate
CLUSTER_ENDPOINT=$(aws eks describe-cluster --region $REGION --name $CLUSTER_NAME --query 'cluster.endpoint' --output text)
CLUSTER_CA=$(aws eks describe-cluster --region $REGION --name $CLUSTER_NAME --query 'cluster.certificateAuthority.data' --output text)

# Create kubelet config
cat > /etc/kubernetes/kubelet/kubelet-config.json <<EOF
{
    "kind": "KubeletConfiguration",
    "apiVersion": "kubelet.config.k8s.io/v1beta1",
    "address": "0.0.0.0",
    "port": 10250,
    "readOnlyPort": 0,
    "cgroupDriver": "systemd",
    "hairpinMode": "hairpin-veth",
    "serializeImagePulls": false,
    "featureGates": {
        "RotateKubeletServerCertificate": true
    },
    "clusterDomain": "cluster.local",
    "clusterDNS": ["172.20.0.10"],
    "resolvConf": "/etc/resolv.conf",
    "runtimeRequestTimeout": "15m",
    "kubeReserved": {
        "cpu": "100m",
        "memory": "100Mi",
        "ephemeral-storage": "1Gi"
    },
    "systemReserved": {
        "cpu": "100m",
        "memory": "100Mi",
        "ephemeral-storage": "1Gi"
    },
    "evictionHard": {
        "memory.available": "100Mi",
        "nodefs.available": "10%",
        "nodefs.inodesFree": "5%"
    },
    "maxPods": 110,
    "authentication": {
        "anonymous": {
            "enabled": false
        },
        "webhook": {
            "enabled": true,
            "cacheTTL": "2m0s"
        }
    },
    "authorization": {
        "mode": "Webhook",
        "webhook": {
            "cacheAuthorizedTTL": "5m0s",
            "cacheUnauthorizedTTL": "30s"
        }
    },
    "registryPullQPS": 10,
    "registryBurst": 20,
    "eventRecordQPS": 5,
    "eventBurst": 10,
    "enableDebuggingHandlers": true,
    "enableContentionProfiling": true,
    "healthzPort": 10248,
    "healthzBindAddress": "127.0.0.1",
    "oomScoreAdj": -999,
    "streamingConnectionIdleTimeout": "4h0m0s",
    "nodeStatusUpdateFrequency": "10s",
    "nodeStatusReportFrequency": "1m0s",
    "nodeLeaseDurationSeconds": 40,
    "imageMinimumGCAge": "2m0s",
    "imageGCHighThresholdPercent": 85,
    "imageGCLowThresholdPercent": 80,
    "volumeStatsAggPeriod": "1m0s",
    "kubeletCgroups": "/systemd/system.slice",
    "systemCgroups": "/systemd/system.slice",
    "cgroupRoot": "/",
    "cgroupsPerQOS": true,
    "enforceNodeAllocatable": ["pods"],
    "runtimeClass": "",
    "topologyManagerPolicy": "none",
    "topologyManagerScope": "container"
}
EOF

# Create kubeconfig
cat > /var/lib/kubelet/kubeconfig <<EOF
apiVersion: v1
kind: Config
clusters:
- cluster:
    certificate-authority-data: $CLUSTER_CA
    server: $CLUSTER_ENDPOINT
  name: kubernetes
contexts:
- context:
    cluster: kubernetes
    user: kubelet
  name: kubelet
current-context: kubelet
users:
- name: kubelet
  user:
    exec:
      apiVersion: client.authentication.k8s.io/v1beta1
      command: /usr/local/bin/aws-iam-authenticator
      args:
        - "token"
        - "-i"
        - "$CLUSTER_NAME"
        - "--region"
        - "$REGION"
EOF

# Get instance metadata
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
LOCAL_IPV4=$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4)
AVAILABILITY_ZONE=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)

# Create kubelet systemd service
cat > /etc/systemd/system/kubelet.service <<EOF
[Unit]
Description=Kubernetes Kubelet
Documentation=https://github.com/kubernetes/kubernetes
After=docker.service
Requires=docker.service

[Service]
ExecStart=/usr/local/bin/kubelet \\
    --config=/etc/kubernetes/kubelet/kubelet-config.json \\
    --kubeconfig=/var/lib/kubelet/kubeconfig \\
    --container-runtime=docker \\
    --image-pull-progress-deadline=2m \\
    --node-ip=$LOCAL_IPV4 \\
    --pod-infra-container-image=602401143452.dkr.ecr.$REGION.amazonaws.com/eks/pause:3.5 \\
    --v=2 \\
    --cloud-provider=aws \\
    --container-runtime-endpoint=unix:///var/run/dockershim.sock \\
    --node-labels=node.kubernetes.io/instance-type=$INSTANCE_TYPE,topology.kubernetes.io/zone=$AVAILABILITY_ZONE,node-type=spot-batch,lifecycle=spot

Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Install and configure AWS Node Termination Handler
if [ "${enable_interruption_handler}" = "true" ]; then
    # Download and install node termination handler
    wget https://github.com/aws/aws-node-termination-handler/releases/download/v1.19.0/node-termination-handler_linux_amd64
    chmod +x node-termination-handler_linux_amd64
    mv node-termination-handler_linux_amd64 /usr/local/bin/node-termination-handler

    # Create systemd service for node termination handler
    cat > /etc/systemd/system/node-termination-handler.service <<EOF
[Unit]
Description=AWS Node Termination Handler
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/node-termination-handler \\
    --node-name=$INSTANCE_ID \\
    --namespace=kube-system \\
    --delete-local-data=true \\
    --ignore-daemon-sets=true \\
    --pod-termination-grace-period=30 \\
    --instance-metadata-url=http://169.254.169.254 \\
    --log-level=info

Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    # Enable and start node termination handler
    systemctl daemon-reload
    systemctl enable node-termination-handler
    systemctl start node-termination-handler
fi

# Configure Docker daemon
cat > /etc/docker/daemon.json <<EOF
{
    "bridge": "none",
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "10"
    },
    "live-restore": true,
    "max-concurrent-downloads": 10,
    "max-concurrent-uploads": 10,
    "storage-driver": "overlay2",
    "storage-opts": [
        "overlay2.override_kernel_check=true"
    ]
}
EOF

# Restart Docker with new configuration
systemctl restart docker

# Install CNI plugins
mkdir -p /opt/cni/bin
wget https://github.com/containernetworking/plugins/releases/download/v1.1.1/cni-plugins-linux-amd64-v1.1.1.tgz
tar -xzf cni-plugins-linux-amd64-v1.1.1.tgz -C /opt/cni/bin/
rm cni-plugins-linux-amd64-v1.1.1.tgz

# Create CNI configuration directory
mkdir -p /etc/cni/net.d

# Install AWS VPC CNI plugin
wget https://github.com/aws/amazon-vpc-cni-k8s/releases/download/v1.12.6/aws-k8s-cni
chmod +x aws-k8s-cni
mv aws-k8s-cni /opt/cni/bin/

# Configure log rotation for kubelet
cat > /etc/logrotate.d/kubelet <<EOF
/var/log/pods/*/*.log {
    rotate 5
    daily
    compress
    missingok
    notifempty
    maxage 30
    copytruncate
}
EOF

# Set up CloudWatch agent for monitoring
yum install -y amazon-cloudwatch-agent

cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<EOF
{
    "agent": {
        "metrics_collection_interval": 60,
        "run_as_user": "cwagent"
    },
    "metrics": {
        "namespace": "EKS/SpotInstances",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    "cpu_usage_idle",
                    "cpu_usage_iowait",
                    "cpu_usage_user",
                    "cpu_usage_system"
                ],
                "metrics_collection_interval": 60,
                "totalcpu": false
            },
            "disk": {
                "measurement": [
                    "used_percent"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "diskio": {
                "measurement": [
                    "io_time"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 60
            },
            "netstat": {
                "measurement": [
                    "tcp_established",
                    "tcp_time_wait"
                ],
                "metrics_collection_interval": 60
            },
            "swap": {
                "measurement": [
                    "swap_used_percent"
                ],
                "metrics_collection_interval": 60
            }
        }
    },
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/messages",
                        "log_group_name": "/aws/eks/$CLUSTER_NAME/spot-batch",
                        "log_stream_name": "{instance_id}/messages"
                    },
                    {
                        "file_path": "/var/log/kubelet/kubelet.log",
                        "log_group_name": "/aws/eks/$CLUSTER_NAME/spot-batch",
                        "log_stream_name": "{instance_id}/kubelet"
                    }
                ]
            }
        }
    }
}
EOF

# Start CloudWatch agent
systemctl enable amazon-cloudwatch-agent
systemctl start amazon-cloudwatch-agent

# Apply node labels and taints
cat > /tmp/label-node.sh <<EOF
#!/bin/bash
# Wait for node to be ready
while ! kubectl get node $INSTANCE_ID; do
    sleep 10
done

# Apply labels
kubectl label node $INSTANCE_ID node-type=spot-batch --overwrite
kubectl label node $INSTANCE_ID lifecycle=spot --overwrite
kubectl label node $INSTANCE_ID instance-type=$INSTANCE_TYPE --overwrite

# Apply taint for spot instances
kubectl taint node $INSTANCE_ID spot=true:NoSchedule --overwrite
EOF

chmod +x /tmp/label-node.sh

# Enable and start kubelet
systemctl daemon-reload
systemctl enable kubelet
systemctl start kubelet

# Wait for kubelet to start and then apply labels
sleep 30
/tmp/label-node.sh &

# Configure automatic cleanup of old Docker images
cat > /etc/cron.d/docker-cleanup <<EOF
# Clean up old Docker images daily at 2 AM
0 2 * * * root docker image prune -f --filter "until=24h"
EOF

# Configure log cleanup
cat > /etc/cron.d/log-cleanup <<EOF
# Clean up old logs daily at 3 AM
0 3 * * * root find /var/log/pods -name "*.log" -mtime +7 -delete
EOF

# Signal completion
/opt/aws/bin/cfn-signal -e $? --stack $CLUSTER_NAME --resource SpotInstanceLaunchTemplate --region $REGION

echo "Spot instance configuration completed successfully"

