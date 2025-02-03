# GraphRAG个人学习笔记

## References
[1] [graphrag的原理&源码及应用介绍](https://www.bilibili.com/video/BV1u6iFeAEx9)  
[2] [https://gitee.com/open-llm/llm-graph-rag](https://gitee.com/open-llm/llm-graph-rag)  
原up主没有把项目上传到github，所以没有fork

## 主要修改
### 原up主的修改
1. 添加了对Qwen大模型的支持
### 本人的修改
1. 解决了windows平台上的中文编解码问题
2. 完善了中文prompts

## 注意
1. 在windows平台上安装时，去掉`requirements.txt`中的`uvloop==0.19.0`  
这个包用于在linux环境下调度协程，不能在windows上安装。  
该包在windows环境下不会被调用，对项目运行无影响。
