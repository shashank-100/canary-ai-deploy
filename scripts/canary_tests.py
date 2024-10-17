#!/usr/bin/env python3
"""
Canary Testing Script for AI Model Serving Platform
Runs integration tests against canary deployments
"""
import asyncio
import argparse
import json
import time
import random
import base64
from typing import Dict, Any, List
import httpx
import logging
from PIL import Image
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CanaryTester:
    """Canary deployment tester"""
    
    def __init__(self, model_name: str, endpoint: str):
        self.model_name = model_name
        self.endpoint = endpoint.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Test data generators
        self.test_generators = {
            "bert-ner": self._generate_bert_test_data,
            "resnet-classifier": self._generate_resnet_test_data
        }
    
    def _generate_bert_test_data(self) -> Dict[str, Any]:
        """Generate test data for BERT NER model"""
        test_texts = [
            "Hello, my name is John Doe and I work at Microsoft in Seattle.",
            "Apple Inc. is headquartered in Cupertino, California.",
            "The meeting is scheduled for Monday at 3 PM in New York.",
            "Barack Obama was the 44th President of the United States.",
            "Google was founded by Larry Page and Sergey Brin in 1998.",
            "The conference will be held in London, United Kingdom.",
            "Tesla's CEO Elon Musk announced new features yesterday.",
            "Amazon Web Services provides cloud computing platforms.",
            "The research was conducted at Stanford University.",
            "Netflix released a new series last Friday."
        ]
        
        return {
            "data": {
                "text": random.choice(test_texts)
            }
        }
    
    def _generate_resnet_test_data(self) -> Dict[str, Any]:
        """Generate test data for ResNet classifier model"""
        # Create a random test image
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        color = random.choice(colors)
        size = random.choice([(64, 64), (128, 128), (224, 224)])
        
        # Create image
        img = Image.new('RGB', size, color=color)
        
        # Add some random noise
        pixels = img.load()
        for i in range(size[0]):
            for j in range(size[1]):
                if random.random() < 0.1:  # 10% noise
                    noise = random.randint(-50, 50)
                    r, g, b = pixels[i, j]
                    pixels[i, j] = (
                        max(0, min(255, r + noise)),
                        max(0, min(255, g + noise)),
                        max(0, min(255, b + noise))
                    )
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "data": {
                "image": img_str
            }
        }
    
    async def test_health_endpoint(self) -> bool:
        """Test health endpoint"""
        try:
            response = await self.client.get(f"{self.endpoint}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def test_prediction_endpoint(self) -> Dict[str, Any]:
        """Test prediction endpoint"""
        if self.model_name not in self.test_generators:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Generate test data
        test_data = self.test_generators[self.model_name]()
        
        start_time = time.time()
        try:
            response = await self.client.post(
                f"{self.endpoint}/predict",
                json=test_data,
                headers={"Content-Type": "application/json"}
            )
            
            latency = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "latency": latency,
                    "status_code": response.status_code,
                    "response": result,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "latency": latency,
                    "status_code": response.status_code,
                    "response": None,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            latency = time.time() - start_time
            return {
                "success": False,
                "latency": latency,
                "status_code": None,
                "response": None,
                "error": str(e)
            }
    
    async def test_batch_prediction_endpoint(self) -> Dict[str, Any]:
        """Test batch prediction endpoint"""
        if self.model_name not in self.test_generators:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Generate batch test data
        batch_size = random.randint(2, 5)
        batch_data = []
        
        for _ in range(batch_size):
            test_data = self.test_generators[self.model_name]()
            batch_data.append(test_data["data"])
        
        request_data = {
            "batch_data": batch_data,
            "batch_size": batch_size
        }
        
        start_time = time.time()
        try:
            response = await self.client.post(
                f"{self.endpoint}/batch-predict",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            latency = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "latency": latency,
                    "status_code": response.status_code,
                    "response": result,
                    "batch_size": batch_size,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "latency": latency,
                    "status_code": response.status_code,
                    "response": None,
                    "batch_size": batch_size,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            latency = time.time() - start_time
            return {
                "success": False,
                "latency": latency,
                "status_code": None,
                "response": None,
                "batch_size": batch_size,
                "error": str(e)
            }
    
    async def run_load_test(self, num_requests: int, concurrency: int = 10) -> Dict[str, Any]:
        """Run load test with specified number of requests"""
        logger.info(f"Starting load test: {num_requests} requests with concurrency {concurrency}")
        
        # First check health
        if not await self.test_health_endpoint():
            return {
                "success": False,
                "error": "Health check failed before load test"
            }
        
        # Prepare semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency)
        
        async def single_test():
            async with semaphore:
                # Randomly choose between single and batch prediction
                if random.random() < 0.8:  # 80% single predictions
                    return await self.test_prediction_endpoint()
                else:  # 20% batch predictions
                    return await self.test_batch_prediction_endpoint()
        
        # Run tests
        start_time = time.time()
        tasks = [single_test() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Process results
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append({
                    "success": False,
                    "error": str(result),
                    "latency": 0
                })
            elif result.get("success", False):
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        # Calculate statistics
        success_count = len(successful_results)
        failure_count = len(failed_results)
        success_rate = success_count / num_requests if num_requests > 0 else 0
        
        latencies = [r["latency"] for r in successful_results]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
        
        throughput = num_requests / total_time if total_time > 0 else 0
        
        # Status codes distribution
        status_codes = {}
        for result in results:
            if not isinstance(result, Exception) and result.get("status_code"):
                code = result["status_code"]
                status_codes[code] = status_codes.get(code, 0) + 1
        
        return {
            "success": success_rate >= 0.95,  # 95% success rate threshold
            "total_requests": num_requests,
            "successful_requests": success_count,
            "failed_requests": failure_count,
            "success_rate": success_rate,
            "total_time": total_time,
            "throughput": throughput,
            "avg_latency": avg_latency,
            "p95_latency": p95_latency,
            "p99_latency": p99_latency,
            "status_codes": status_codes,
            "errors": [r.get("error") for r in failed_results if r.get("error")]
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Canary deployment tester")
    parser.add_argument("--model", required=True, help="Model name (bert-ner, resnet-classifier)")
    parser.add_argument("--endpoint", required=True, help="Model endpoint URL")
    parser.add_argument("--tests", type=int, default=100, help="Number of test requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Create tester
    tester = CanaryTester(args.model, args.endpoint)
    
    try:
        # Run tests
        logger.info(f"Testing {args.model} at {args.endpoint}")
        results = await tester.run_load_test(args.tests, args.concurrency)
        
        # Print results
        logger.info("Test Results:")
        logger.info(f"  Total Requests: {results['total_requests']}")
        logger.info(f"  Successful: {results['successful_requests']}")
        logger.info(f"  Failed: {results['failed_requests']}")
        logger.info(f"  Success Rate: {results['success_rate']:.2%}")
        logger.info(f"  Throughput: {results['throughput']:.2f} req/s")
        logger.info(f"  Avg Latency: {results['avg_latency']:.3f}s")
        logger.info(f"  P95 Latency: {results['p95_latency']:.3f}s")
        logger.info(f"  P99 Latency: {results['p99_latency']:.3f}s")
        
        if results['status_codes']:
            logger.info("  Status Codes:")
            for code, count in results['status_codes'].items():
                logger.info(f"    {code}: {count}")
        
        if results.get('errors'):
            logger.warning("  Errors:")
            for error in set(results['errors']):
                logger.warning(f"    {error}")
        
        # Save results to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        
        # Exit with appropriate code
        if results['success']:
            logger.info("✅ Canary tests PASSED")
            exit(0)
        else:
            logger.error("❌ Canary tests FAILED")
            exit(1)
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        exit(1)
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

