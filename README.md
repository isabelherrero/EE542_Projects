# EE 542: Internet and Cloud Computing Projects

This repository contains coursework completed for **EE 542: Internet and Cloud Computing** at the University of Southern California. The labs explore network programming, distributed systems, cloud and IoT platforms, GPU acceleration, machine learning, high-performance networking, and retrieval-augmented generation.

## Labs

| Lab | Topic | Highlights |
| --- | --- | --- |
| [Lab 2](./ee542_lab2/) | Reliable UDP File Transfer | C++ client-server file transfer with NACK-based selective retransmission, multithreading, path-MTU-aware payloads, and MD5 verification. |
| [Lab 3](./ee542_lab3/) | Network Performance Analysis | Network measurement collection and Python-based visualization of experimental results. |
| [Lab 4](./ee542_lab4/) | Hadoop and Apache Spark | Distributed word count, character count, and minimum/maximum analysis using Hadoop MapReduce and Spark. |
| [Lab 5](./ee542_lab5/) | ThingsBoard IoT Application | REST API application that retrieves user and USC library locations, calculates the group center, and recommends the closest library. |
| [Lab 6](./ee542_lab6/) | CUDA and GPU Computing | CPU and CUDA matrix multiplication, tiled kernels, cuBLAS, optimization techniques, Python/CUDA shared libraries, and CPU/GPU image convolution. |
| [Lab 7](./ee542_lab7/) | Neural Networks and CNNs | NumPy implementations of regression, classification, multilayer perceptrons, and CNNs, followed by PyTorch CNN training and GPU benchmarking. |
| [Lab 8](./ee542_lab8/) | Accelerated Networking and Inference | TensorRT inference, packet generation, DPDK receive processing, image classification, GPUNetIO, and CPU/GPU RDMA experiments. |
| [Lab 8 - Part 7](./ee542_lab8_part7/) | GPUNetIO and RDMA Evaluation | CloudLab setup scripts and benchmark results for GPU-to-GPU RDMA latency and bandwidth, CPU RDMA baselines, and DOCA UDP receive tests. |
| [Lab 9](./ee542_lab9/) | LLMs, Embeddings, and RAG | Retrieval-augmented generation using OpenAI embeddings, FAISS indexes, custom histogram/Word2Vec embeddings, and a chatbot backend. |

## Technologies

- **Languages:** C, C++, CUDA C/C++, Python, Bash
- **Networking:** TCP/IP, UDP, sockets, iPerf, DPDK, RDMA, GPUNetIO, NVIDIA DOCA
- **Cloud and distributed computing:** AWS, CloudLab, Hadoop, HDFS, Apache Spark, ThingsBoard
- **GPU and machine learning:** CUDA, cuBLAS, NumPy, PyTorch, TensorRT, convolutional neural networks
- **AI and data:** OpenAI embeddings, FAISS, Word2Vec, retrieval-augmented generation

## Repository Structure

```text
EE542_Projects/
├── ee542_lab2/        # Reliable UDP file transfer
├── ee542_lab3/        # Network measurements and plotting
├── ee542_lab4/        # Hadoop MapReduce and Spark
├── ee542_lab5/        # ThingsBoard nearest-library application
├── ee542_lab6/        # CUDA, cuBLAS, optimization, and convolution
├── ee542_lab7/        # NumPy and PyTorch neural networks
├── ee542_lab8/        # TensorRT, DPDK, classification, and GPUNetIO
├── ee542_lab8_part7/  # CloudLab RDMA scripts and benchmark results
└── ee542_lab9/        # Embeddings, FAISS, RAG, and chatbot code
```

Each lab directory contains the relevant source code, results, and available setup or execution instructions. Some labs require specialized environments such as AWS GPU instances, Hadoop/Spark clusters, ThingsBoard, or CloudLab nodes with NVIDIA and RDMA-capable hardware.

## Selected Project Results

- The Lab 2 `zap` protocol transferred a 1 GB file at **58 Mbps** under 200 ms RTT and 20% packet loss, and **86 Mbps** under 10 ms RTT and 1% packet loss on emulated 100 Mbps links.
- Lab 6 compares progressively optimized CUDA matrix-multiplication kernels with CPU and cuBLAS implementations and applies CUDA acceleration to image convolution.
- Lab 7 includes both from-scratch NumPy CNN work and PyTorch GPU training results.
- Lab 8 evaluates accelerated packet processing, inference, and communication paths across CPU, GPU, DPDK, RDMA, and GPUNetIO workflows.
- Lab 9 compares hosted and custom embedding approaches for semantic retrieval and integrates the resulting FAISS index into a RAG chatbot pipeline.

## Notes

This repository is intended as an academic portfolio. Environment-specific credentials, datasets, and large external dependencies are not included. See the README inside each lab folder for any available requirements and commands.
