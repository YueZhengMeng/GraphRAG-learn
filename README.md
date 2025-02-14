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
3. 在执行实体抽取时，可以通过配置`max_gleanings`进行多次提取，确保提取出尽可能多的实体。  
   但我通过调试过程发现，在重复提取时，只有`第一次实体提取的结果和要求补充提取的prompt`会被封装在`messages`中发送给LLM，`第一次发送给LLM的原始文本和实体提取prompt`并不包括在其中。  
   分析现有代码发现，执行调用LLM的函数`graphrag/llm/openai/openai_chat_llm.py/_execute_llm()`并不会将`input`放入`history`中，且只返回`模型的输出的content`。`content`会被装饰在`_execute_llm()`外层的`graphrag/llm/openai/openai_history_tracking_llm.py/OpenAIHistoryTrackingLLM`类型的组件接收，然后以`system role`放置在`history`当中。  
   实测发现，当前的重复提取逻辑，会导致LLM出现幻觉。为避免该问题，且加快执行速度、减少LLM的调用次数与token消耗，我选择设置`max_gleanings=0`，即不重复提取。  
   在`settings.yaml`中设置`max_gleanings=0`无效，原因是`graphrag/config/create_graphrag_config.py`中420行疑似存在bug，在读取到的0与默认值1之间取了or，导致值一定为1  
   所以我手动在实体提取工作流文件`graphrag/index/workflows/v1/create_base_extracted_entities.py`中修改配置，使得`max_gleanings`为0。  
4. 我认为，在`create_final_entities`这个工作流中对实体节点信息进行embedding时，应该对实体名字段`name`和描述字段`description`都进行embedding，便于之后检索阶段的进行基于语义近似的模糊匹配。  
   但是默认配置只对实体描述字段`description`进行embedding，跳过了对实体名字段`name`的embedding。而且我找了很久也没找到该配置应该在哪里修改。    
   所以我在`graphrag/index/workflows/v1/create_final_entities.py`中修改配置，手动设置`skip_name_embedding`和`skip_description_embedding`都为`False`。  
5. 在`create_final_relationships`工作流中，也通过手动修改`graphrag/index/workflows/v1/create_final_relationships.py`中的配置，启动了对边的描述字段`description`的embedding。  
6. 在`create_final_community_reports`工作流中，也通过手动修改`graphrag/index/workflows/v1/create_final_community_reports.py`中的配置，启动了对社区报告标题、内容和全文的embedding。  
7. 在`create_final_text_units`工作流中，也通过手动修改`graphrag/index/workflows/v1/create_final_text_units.py`中的配置，启动了对文本片段的embedding。  
8. 在`create_final_documents`工作流中，也通过手动修改`graphrag/index/workflows/v1/create_final_documents.py`中的配置，启动了原始文档的embedding。  

## 注意
1. 在windows平台上安装时，去掉`requirements.txt`中的`uvloop==0.19.0`  
这个包用于在linux环境下调度协程，不能在windows上安装。  
该包在windows环境下不会被调用，对项目运行无影响。
