# GraphRAG个人学习笔记

## References
[1] [graphrag的原理&源码及应用介绍](https://www.bilibili.com/video/BV1u6iFeAEx9)  
[2] [https://gitee.com/open-llm/llm-graph-rag](https://gitee.com/open-llm/llm-graph-rag)  
原up主没有把项目上传到github，所以没有fork

## 主要修改
### 原up主的修改
1. 添加了对Qwen等国产大模型的支持
2. 翻译了部分prompts为中文
3. 为部分关键步骤添加了注释
### 本人的修改
1. 解决了windows平台上的中文编解码问题
2. 完善了中文prompts
3. 为几乎所有核心逻辑添加了较为详细的注释
4. 在执行实体抽取时，可以通过配置`max_gleanings`进行多次提取，确保提取出尽可能多的实体。  
   但我通过调试过程发现，在重复提取时，只有`第一次实体提取的结果和要求补充提取的prompt`会被封装在`messages`中发送给LLM，`第一次发送给LLM的原始文本和实体提取prompt`并不包括在其中。  
   分析现有代码发现，执行调用LLM的函数`graphrag/llm/openai/openai_chat_llm.py/_execute_llm()`并不会将`input`放入`history`中，且只返回`模型的输出的content`。`content`会被装饰在`_execute_llm()`外层的`graphrag/llm/openai/openai_history_tracking_llm.py/OpenAIHistoryTrackingLLM`类型的组件接收，然后以`system role`放置在`history`当中。  
   实测发现，当前的重复提取逻辑，会导致LLM出现幻觉。为避免该问题，且加快执行速度、减少LLM的调用次数与token消耗，我选择设置`max_gleanings=0`，即不重复提取。  
   在`settings.yaml`中设置`max_gleanings=0`无效，原因是`graphrag/config/create_graphrag_config.py`中420行疑似存在bug，在读取到的0与默认值1之间取了or，导致值一定为1  
   所以我手动在实体提取工作流文件`graphrag/index/workflows/v1/create_base_extracted_entities.py`中修改配置，使得`max_gleanings`为0。  
5. 我认为，在`create_final_entities`这个工作流中对实体节点信息进行embedding时，应该对实体名字段`name`和描述字段`description`都进行embedding，便于之后检索阶段的进行基于语义近似的模糊匹配。  
   但是默认配置只对实体描述字段`description`进行embedding，跳过了对实体名字段`name`的embedding。而且我找了很久也没找到该配置应该在哪里修改。    
   所以我在`graphrag/index/workflows/v1/create_final_entities.py`中修改配置，手动设置`skip_name_embedding`和`skip_description_embedding`都为`False`。  
6. 在`create_final_relationships`工作流中，也通过手动修改`graphrag/index/workflows/v1/create_final_relationships.py`中的配置，启动了对边的描述字段`description`的embedding。  
7. 在`create_final_community_reports`工作流中，也通过手动修改`graphrag/index/workflows/v1/create_final_community_reports.py`中的配置，启动了对社区报告标题、内容和全文的embedding。  
8. 在`create_final_text_units`工作流中，也通过手动修改`graphrag/index/workflows/v1/create_final_text_units.py`中的配置，启动了对文本片段的embedding。  
9. 在`create_final_documents`工作流中，也通过手动修改`graphrag/index/workflows/v1/create_final_documents.py`中的配置，启动了原始文档的embedding。  
10. 我也不知道为什么，在`settings.yaml`中修改`local_search:top_k_mapped_entities`无效，但是修改`local_search:top_k_relationships`有效。  
    该参数用于控制本地检索时，query与entity description的向量检索返回的实体个数。  
    此外，需要注意：相关源码中设置了源码里设置了`oversample_scaler=2`，即实际返回的实体个数是`oversample_scaler * top_k_mapped_entities`  

## 注意
1. 在windows平台上安装时，去掉`requirements.txt`中的`uvloop==0.19.0`  
这个包用于在linux环境下调度协程，不能在windows上安装。  
该包在windows环境下不会被调用，对项目运行无影响。

# 项目使用说明

## 环境准备
```shell
# 1.安装anaconda
   # 下载anacona，可以下载免费了，填入邮箱，然后通过邮箱收到的链接下载
   # 地址：https://www.anaconda.com/download
   
# 2.创建python环境
conda create -n graphrag python=3.10

# 3.激活环境
conda activate graphrag

# 4. 下载本项目
git clone https://github.com/YueZhengMeng/graphrag-learn
cd graphrag-learn

# 5. 安装graphrag所需环境
# 注意：windows平台安装去掉uvloop这个包
pip install -r requirements.txt
```

## 实例展示

### 准备数据集
《孔乙己》小说原文已存储在`data/kongyiji.txt`中，该文件为utf-8编码

### 初始化项目文件夹
```shell
mkdir -p ./kongyiji
python -m graphrag.index --init --root ./kongyiji
```

### 准备文本数据
```shell
mkdir -p ./kongyiji/input
cp data/kongyiji.txt ./kongyiji/input/
```

### 复制配置文件与中文prompts
```shell
cp extra_data/settings.yaml ./kongyiji
cp extra_data/claim_extraction_cn.txt ./kongyiji/prompts
cp extra_data/community_report_cn.txt ./kongyiji/prompts
cp extra_data/entity_extraction_cn.txt ./kongyiji/prompts
cp extra_data/global_map_system_prompt_cn.txt ./kongyiji/prompts
cp extra_data/global_reduce_system_prompt_cn.txt ./kongyiji/prompts
cp extra_data/local_query_prompt_cn.txt ./kongyiji/prompts
cp extra_data/summarize_descriptions_cn.txt ./kongyiji/prompts
```

### 构建索引
配置resume参数之后，失败了重新跑一遍就可以从断点继续，避免从头开始
```shell
python -m graphrag.index --root ./kongyiji --resume kongyiji -v
```

### 查询
通过--data指定读取哪个索引，否则会读取`kongyiji`目录下的第一个版本
#### 本地查询
```shell
python -m graphrag.query --root ./kongyiji --data ./kongyiji/output/kongyiji/artifacts --method local "孔乙己与丁举人的关系"
```
#### 全局查询
```shell
python -m graphrag.query --root ./kongyiji --data ./kongyiji/output/kongyiji/artifacts --method global "孔乙己与丁举人的关系"
```

### 知识图谱可视化
基于`pyvis`实现  
代码见`./visualize_graph/visualize_graph.ipynb`  
可视化结果可以直接在`Jupyter Notebook`界面查看，也可以用浏览器打开生成的html文件  
界面第一次显示有点慢，推测与某些必要的css与js文件在外网有关  
第一次显示后，当前路径下会出现`lib`文件夹，其中缓存了这些css与js文件  
确保`lib`文件夹与涉及`pyvis`的`ipynb`文件或`html`文件在同一文件夹下，之后的页面会自动加载这些文件，速度会变快  
鼠标放在节点或边上，会显示对应的详细信息  
节点可以拖动，拖动结束后会自动调整位置与图的形状

## 个人感想与建议
微软的程序员们，在这个项目中，展示了其登峰造极的软件工程学水平。  
但显然，他们没有尝试站在初学者的角度，考虑学习阶梯与成本的问题。  
此外，该项目的早期实验性质也很明显，配置文件、代码注释和文档，都存在巨大的完善空间。  
以上并非是我一个人的观点，另一个GraphRAG的精简实现项目[nano-graphrag](https://github.com/gusye1234/nano-graphrag)的作者也给出了近似的评价。  
  
GraphRAG的核心算法，其实并不复杂，用两页PPT展示其中大约5个关键步骤就能够讲明白。  
索引构建部分，阅读本项目中`./visualize_graph/visualize_graph.ipynb`文件展示的索引构建结果、可视化结果、以及说明性注释即可。  
查询部分，从`graphrag/query/cli.py`文件开始，阅读`LocalSearch`与`GlobalSearch`两种搜索引擎的`search`方法的源码与注释，尤其是其中的`context builder`的实现即可。  
  
综上，我个人建议大家直接阅读[nano-graphrag](https://github.com/gusye1234/nano-graphrag)项目。   
一方面是因为[nano-graphrag](https://github.com/gusye1234/nano-graphrag)只用几百行简单直白的代码就实现了GraphRAG的核心算法，另一方面是因为更适合应用于实际生产场景的[LightRAG](https://github.com/HKUDS/LightRAG)框架也是基于[nano-graphrag](https://github.com/gusye1234/nano-graphrag)二次开发的。  
我之后也会阅读和解析以上两个项目，并分析自己的学习笔记。
