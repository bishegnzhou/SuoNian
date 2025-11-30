# AI Researcher (DeepSeek 本地版)

本项目基于 Autonomous Researcher 深度优化，专为 **DeepSeek V3** 和 **本地运行** 打造。

**核心特性：**
*   **DeepSeek 集成**：完美支持 DeepSeek V3，优化中文提示词，解决无原生工具调用问题。
*   **完全本地化**：无需 Modal 云环境，直接利用本地算力，零成本运行。
*   **一键转文档**：内置 `SuoNian.py`，一键将 Markdown 报告转为精美 Word 文档。
*   **中文适配**：修复 Windows 乱码，强制输出简体中文报告。

**快速开始：**
1.  **安装**：`pip install -r requirements.txt`
2.  **配置**：在 `.env` 中填入 `DEEPSEEK_API_KEY=sk-...`
3.  **启动**：运行 `python run_app.py`，浏览器访问 `http://localhost:5173`

**使用流程：**
网页输入课题 -> 自动研究 -> 生成 `final_report.md` -> 运行 `python SuoNian.py` 导出 Word。

*致谢：原项目来自 [mshumer](https://github.com/mshumer/autonomous-researcher)*
