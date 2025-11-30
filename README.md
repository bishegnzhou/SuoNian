# AI Researcher (DeepSeek 深度优化版)

> **致谢 / Attribution**
> 本项目基于 [mshumer/autonomous-researcher](https://github.com/mshumer/autonomous-researcher) 二次开发。
> 感谢原作者 Matt Shumer 的开源贡献。

## 📖 项目介绍

这是一个全自动的 AI 研究员助手。它能根据您给出的研究课题，自动拆解任务、联网搜索资料、阅读分析文献，最终生成一份专业的**中文研究报告**。

**本项目针对 DeepSeek V3 和本地环境进行了深度优化：**

*   🚀 **DeepSeek 驱动**：针对 DeepSeek V3 模型优化了提示词，确保输出高质量的简体中文报告。
*   💻 **完全本地化**：默认支持本地运行（Local Sandbox），无需配置昂贵的 Modal 云端环境，零成本部署。
*   📄 **一键导出 Word**：内置格式转换工具，可将生成的 Markdown 报告一键转换为排版精美的 Word 文档。
*   🔧 **中文适配**：修复了 Windows 系统下的编码乱码问题，优化了中文显示体验。

## 🚀 快速开始

### 1. 安装依赖
确保安装了 Python 3.10+，然后在终端运行：
```bash
pip install -r requirements.txt
```

### 2. 配置密钥
在项目根目录新建 `.env` 文件，填入您的 DeepSeek API Key：
```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. 启动应用
运行以下命令，一键启动后端和前端：
```bash
python run_app.py
```
启动成功后，浏览器会自动打开 `http://localhost:5173`。

## 📝 使用指南

1.  **开始研究**：在网页输入框输入课题（如“2024年人工智能发展趋势”），点击 Start Research。
2.  **获取报告**：研究完成后，根目录会生成 `final_report.md`。
3.  **导出文档**：在终端运行 `python SuoNian.py`，即可获得 `final_report.docx`。