# Local_QA_based_on_ChatGLM
基于ChatGLM大语言模型构建的中文本地问答系统

## 前言
使用RAG针对ChatGLM大语言模型做了一个简单的问答系统，embedding模型是text2vec-base-chinese，并使用Gradio构建了一个简单的对话系统。

## 功能
- 可以读取pdf、doc(或docx)、txt文件并进行embedding
- 当提出问题时，提取相关的文章段落
- 根据文章段落，进行提示词工程，模型会根据提示词回答问题并给出问题答案（这一步骤对用户透明，即提示词工程自动完成）
## 模型下载地址

### huggingface地址：
- text2vec-base-chinese：https://huggingface.co/shibing624/text2vec-base-chinese
- chatglm-6b：https://huggingface.co/THUDM/chatglm-6b

## 网盘下载
如果huggingface下载有问题，可以网盘下载
链接：https://pan.baidu.com/s/1WnTGlo0LI9xkT4-Hufr4Kw?pwd=q2j4
提取码：q2j4

## 运行效果展示
哔哩哔哩： https://www.bilibili.com/video/BV1tw4m1k7jG
