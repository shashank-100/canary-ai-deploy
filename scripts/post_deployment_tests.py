#!/usr/bin/env python3
"""
Post-Deployment Testing Script for AI Model Serving Platform
Comprehensive tests after deployment completion
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


class PostDeploymentTester:
    """Comprehensive post-deployment tester"""
    
    def __init__(self, model_name: str, endpoint: str):
        self.model_name = model_name
        self.endpoint = endpoint.rstrip('/')
        self.client = httpx.AsyncClient(timeout=60.0)
        
        # Test configurations
        self.test_suites = {
            "health": self._test_health_endpoints,
            "functionality": self._test_functionality,
            "performance": self._test_performance,
            "reliability": self._test_reliability,
            "security": self._test_security
        }
    
    async def _test_health_endpoints(self) -> Dict[str, Any]:
        """Test health and status endpoints"""
        results = {}
        
        # Health check
        try:
            response = await self.client.get(f"{self.endpoint}/health")
            results["health_check"] = {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "response": response.json() if response.status_code == 200 else response.text
            }
        except Exception as e:
            results["health_check"] = {
                "success": False,
                "error": str(e)
            }
        
        # Readiness check
        try:
            response = await self.client.get(f"{self.endpoint}/ready")
            results["readiness_check"] = {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "response": response.json() if response.status_code == 200 else response.text
            }
        except Exception as e:
            results["readiness_check"] = {
                "success": False,
                "error": str(e)
            }
        
        # Metrics endpoint
        try:
            response = await self.client.get(f"{self.endpoint}/metrics")
            results["metrics_check"] = {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "has_prometheus_metrics": "model_requests_total" in response.text if response.status_code == 200 else False
            }
        except Exception as e:
            results["metrics_check"] = {
                "success": False,
                "error": str(e)
            }
        
        # Model info endpoint
        try:
            response = await self.client.get(f"{self.endpoint}/model/info")
            results["model_info_check"] = {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "response": response.json() if response.status_code == 200 else response.text
            }
        except Exception as e:
            results["model_info_check"] = {
                "success": False,
                "error": str(e)
            }
        
        return results
    
    async def _test_functionality(self) -> Dict[str, Any]:
        """Test core functionality"""
        results = {}
        
        # Single prediction test
        try:
            test_data = self._generate_test_data()
            response = await self.client.post(
                f"{self.endpoint}/predict",
                json=test_data,
                headers={"Content-Type": "application/json"}
            )
            
            results["single_prediction"] = {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "has_predictions": "predictions" in response.json() if response.status_code == 200 else False,
                "response": response.json() if response.status_code == 200 else response.text
            }
        except Exception as e:
            results["single_prediction"] = {
                "success": False,
                "error": str(e)
            }
        
        # Batch prediction test
        try:
            batch_data = {
                "batch_data": [self._generate_test_data()["data"] for _ in range(3)],
                "batch_size": 3
            }
            response = await self.client.post(
                f"{self.endpoint}/batch-predict",
                json=batch_data,
                headers={"Content-Type": "application/json"}
            )
            
            results["batch_prediction"] = {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "has_predictions": "predictions" in response.json() if response.status_code == 200 else False,
                "response": response.json() if response.status_code == 200 else response.text
            }
        except Exception as e:
            results["batch_prediction"] = {
                "success": False,
                "error": str(e)
            }
        
        # Model-specific functionality tests
        if self.model_name == "bert-ner":
            results.update(await self._test_bert_functionality())
        elif self.model_name == "resnet-classifier":
            results.update(await self._test_resnet_functionality())
        
        return results
    
    async def _test_bert_functionality(self) -> Dict[str, Any]:
        """Test BERT NER specific functionality"""
        results = {}
        
        # Test with various text inputs
        test_cases = [
            "John Doe works at Microsoft.",
            "Apple Inc. is in Cupertino.",
            "Meeting on Monday in New York.",
            "",  # Empty text
            "No entities here.",
            "Multiple entities: Barack Obama, Google, Stanford University, Amazon."
        ]
        
        for i, text in enumerate(test_cases):
            try:
                response = await self.client.post(
                    f"{self.endpoint}/predict",
                    json={"data": {"text": text}},
                    headers={"Content-Type": "application/json"}
                )
                
                results[f"bert_test_case_{i}"] = {
                    "input": text,
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds(),
                    "response": response.json() if response.status_code == 200 else response.text
                }
            except Exception as e:
                results[f"bert_test_case_{i}"] = {
                    "input": text,
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    async def _test_resnet_functionality(self) -> Dict[str, Any]:
        """Test ResNet classifier specific functionality"""
        results = {}
        
        # Test with various image inputs
        test_cases = [
            {"size": (224, 224), "color": (255, 0, 0)},    # Red image
            {"size": (128, 128), "color": (0, 255, 0)},    # Green image
            {"size": (64, 64), "color": (0, 0, 255)},      # Blue image
            {"size": (32, 32), "color": (255, 255, 255)},  # White image
        ]
        
        for i, case in enumerate(test_cases):
            try:
                # Create test image
                img = Image.new('RGB', case["size"], color=case["color"])
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                
                response = await self.client.post(
                    f"{self.endpoint}/predict",
                    json={"data": {"image": img_str}},
                    headers={"Content-Type": "application/json"}
                )
                
                results[f"resnet_test_case_{i}"] = {
                    "input": f"{case['size']} {case['color']} image",
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds(),
                    "response": response.json() if response.status_code == 200 else response.text
                }
            except Exception as e:
                results[f"resnet_test_case_{i}"] = {
                    "input": f"{case['size']} {case['color']} image",
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    async def _test_performance(self) -> Dict[str, Any]:
        """Test performance characteristics"""
        results = {}
        
        # Latency test
        latencies = []
        for _ in range(10):
            start_time = time.time()
            try:
                test_data = self._generate_test_data()
                response = await self.client.post(
                    f"{self.endpoint}/predict",
                    json=test_data,
                    headers={"Content-Type": "application/json"}
                )
                latency = time.time() - start_time
                if response.status_code == 200:
                    latencies.append(latency)
            except Exception:
                pass
        
        if latencies:
            results["latency_test"] = {
                "success": True,
                "avg_latency": sum(latencies) / len(latencies),
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)],
                "samples": len(latencies)
            }
        else:
            results["latency_test"] = {
                "success": False,
                "error": "No successful requests"
            }
        
        # Concurrent requests test
        async def single_request():
            try:
                test_data = self._generate_test_data()
                response = await self.client.post(
                    f"{self.endpoint}/predict",
                    json=test_data,
                    headers={"Content-Type": "application/json"}
                )
                return response.status_code == 200
            except Exception:
                return False
        
        start_time = time.time()
        concurrent_tasks = [single_request() for _ in range(20)]
        concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        concurrent_time = time.time() - start_time
        
        successful_concurrent = sum(1 for r in concurrent_results if r is True)
        
        results["concurrency_test"] = {
            "success": successful_concurrent >= 18,  # 90% success rate
            "total_requests": 20,
            "successful_requests": successful_concurrent,
            "success_rate": successful_concurrent / 20,
            "total_time": concurrent_time,
            "throughput": 20 / concurrent_time
        }
        
        return results
    
    async def _test_reliability(self) -> Dict[str, Any]:
        """Test reliability and error handling"""
        results = {}
        
        # Invalid input test
        try:
            response = await self.client.post(
                f"{self.endpoint}/predict",
                json={"invalid": "data"},
                headers={"Content-Type": "application/json"}
            )
            
            results["invalid_input_test"] = {
                "success": response.status_code in [400, 422],  # Should return client error
                "status_code": response.status_code,
                "response": response.text
            }
        except Exception as e:
            results["invalid_input_test"] = {
                "success": False,
                "error": str(e)
            }
        
        # Large payload test
        try:
            if self.model_name == "bert-ner":
                large_text = "This is a test. " * 1000  # Large text
                test_data = {"data": {"text": large_text}}
            else:
                # Create large image
                img = Image.new('RGB', (1024, 1024), color=(128, 128, 128))
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                test_data = {"data": {"image": img_str}}
            
            response = await self.client.post(
                f"{self.endpoint}/predict",
                json=test_data,
                headers={"Content-Type": "application/json"}
            )
            
            results["large_payload_test"] = {
                "success": response.status_code in [200, 413, 422],  # Success or appropriate error
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            results["large_payload_test"] = {
                "success": False,
                "error": str(e)
            }
        
        # Stress test
        stress_results = []
        for _ in range(50):
            try:
                test_data = self._generate_test_data()
                response = await self.client.post(
                    f"{self.endpoint}/predict",
                    json=test_data,
                    headers={"Content-Type": "application/json"}
                )
                stress_results.append(response.status_code == 200)
            except Exception:
                stress_results.append(False)
        
        success_rate = sum(stress_results) / len(stress_results)
        results["stress_test"] = {
            "success": success_rate >= 0.95,  # 95% success rate
            "total_requests": len(stress_results),
            "successful_requests": sum(stress_results),
            "success_rate": success_rate
        }
        
        return results
    
    async def _test_security(self) -> Dict[str, Any]:
        """Test security aspects"""
        results = {}
        
        # CORS test
        try:
            response = await self.client.options(
                f"{self.endpoint}/predict",
                headers={"Origin": "https://example.com"}
            )
            
            results["cors_test"] = {
                "success": "Access-Control-Allow-Origin" in response.headers,
                "status_code": response.status_code,
                "cors_headers": {k: v for k, v in response.headers.items() if k.startswith("Access-Control")}
            }
        except Exception as e:
            results["cors_test"] = {
                "success": False,
                "error": str(e)
            }
        
        # Content-Type validation
        try:
            response = await self.client.post(
                f"{self.endpoint}/predict",
                data="invalid json",
                headers={"Content-Type": "text/plain"}
            )
            
            results["content_type_test"] = {
                "success": response.status_code in [400, 415, 422],  # Should reject invalid content type
                "status_code": response.status_code
            }
        except Exception as e:
            results["content_type_test"] = {
                "success": False,
                "error": str(e)
            }
        
        return results
    
    def _generate_test_data(self) -> Dict[str, Any]:
        """Generate test data based on model type"""
        if self.model_name == "bert-ner":
            test_texts = [
                "Hello, my name is John Doe and I work at Microsoft.",
                "Apple Inc. is headquartered in Cupertino, California.",
                "The meeting is scheduled for Monday in New York."
            ]
            return {"data": {"text": random.choice(test_texts)}}
        
        elif self.model_name == "resnet-classifier":
            # Create a simple test image
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            color = random.choice(colors)
            img = Image.new('RGB', (64, 64), color=color)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return {"data": {"image": img_str}}
        
        else:
            return {"data": {"test": "data"}}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites"""
        logger.info(f"Starting comprehensive tests for {self.model_name}")
        
        all_results = {}
        overall_success = True
        
        for suite_name, test_function in self.test_suites.items():
            logger.info(f"Running {suite_name} tests...")
            
            try:
                suite_results = await test_function()
                all_results[suite_name] = suite_results
                
                # Check if suite passed
                suite_success = all(
                    result.get("success", False) 
                    for result in suite_results.values() 
                    if isinstance(result, dict)
                )
                
                if suite_success:
                    logger.info(f"✅ {suite_name} tests PASSED")
                else:
                    logger.warning(f"⚠️ {suite_name} tests had failures")
                    overall_success = False
                    
            except Exception as e:
                logger.error(f"❌ {suite_name} tests FAILED: {e}")
                all_results[suite_name] = {"error": str(e)}
                overall_success = False
        
        # Calculate summary statistics
        total_tests = 0
        passed_tests = 0
        
        for suite_results in all_results.values():
            if isinstance(suite_results, dict) and "error" not in suite_results:
                for result in suite_results.values():
                    if isinstance(result, dict) and "success" in result:
                        total_tests += 1
                        if result["success"]:
                            passed_tests += 1
        
        summary = {
            "overall_success": overall_success,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "test_results": all_results
        }
        
        return summary
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Post-deployment comprehensive tester")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--endpoint", required=True, help="Model endpoint URL")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Create tester
    tester = PostDeploymentTester(args.model, args.endpoint)
    
    try:
        # Run all tests
        logger.info(f"Testing {args.model} deployment at {args.endpoint}")
        results = await tester.run_all_tests()
        
        # Print summary
        logger.info("Test Summary:")
        logger.info(f"  Overall Success: {results['overall_success']}")
        logger.info(f"  Total Tests: {results['total_tests']}")
        logger.info(f"  Passed: {results['passed_tests']}")
        logger.info(f"  Failed: {results['failed_tests']}")
        logger.info(f"  Success Rate: {results['success_rate']:.2%}")
        
        # Save results to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        
        # Exit with appropriate code
        if results['overall_success']:
            logger.info("✅ Post-deployment tests PASSED")
            exit(0)
        else:
            logger.error("❌ Post-deployment tests FAILED")
            exit(1)
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        exit(1)
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

