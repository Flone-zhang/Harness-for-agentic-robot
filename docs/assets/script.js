"use strict";
(() => {
  const nodes = Array.from(document.querySelectorAll("[data-i18n]"));
  const english = Object.fromEntries(nodes.map(node => [node.dataset.i18n, node.textContent]));
  const chinese = {
    skip: "跳到正文", navOverview: "概述", navDemo: "演示", navMethod: "方法", navResults: "结果",
    heroTag: "长时序机器人操作", draftTag: "V03 · 研究草稿",
    title: "面向长时序机器人操作的有状态分层 Harness：动态记忆与 Watchdog 恢复机制",
    heroDescription: "以固定的 π0 为执行器，用显式任务状态与闭环运行时协调规划、记忆、验证和恢复。",
    authors: "匿名作者 · V03 论文草稿", paperButton: "论文 · Word（英文）", codeButton: "代码", demoButton: "观看演示",
    stat1: "17 / 20 次试验成功", stat2: "相对单体 π0 的提升", stat3: "实体机器人实验", pp: "百分点",
    enlarge: "查看原图 ↗",
    architectureCaption: "多模态 Agent、动态记忆、已注册 π0 技能与 Watchdog 反馈共同构成有状态执行闭环。",
    overviewLabel: "01 / 研究概述", overviewTitle: "长时序任务需要持续维护状态的运行时。",
    abstract1: "视觉—语言—动作（VLA）策略将语言和图像映射为连续机器人动作，但长时序执行仍容易受到误差累积、子任务边界隐式化和任务状态丢失的影响。",
    abstract2: "HarnessVLA 将固定 π0 执行器与任务分解、动态记忆、类型化技能调用和 Watchdog 监控分离。每次决策前，Agent 都会组装当前观测、任务状态、已注册工具和经过有效性过滤的持久事实。",
    abstract3: "在实体 Piper 机械臂上，完整系统成功完成 17/20 次试验（85%），单体 π0 为 3/20（15%）。独立组件比较中，子任务分解带来了最大的已观察单组件提升。",
    demoLabel: "02 / 真机演示", demoTitle: "一条指令，连续完成多个动作。",
    demoDescription: "将红色积木放入蓝色盒子，将绿色积木放入碗中，然后按下红色按钮。",
    videoFallback: "浏览器无法播放此视频，请使用下方下载链接。",
    videoCaption: "论文配套的 T1 剪辑演示。视频展示执行过程；下方成功率来自论文报告的试验次数。",
    downloadVideo: "下载视频 ↓", videoError: "视频加载失败，请尝试下载后播放。",
    methodLabel: "03 / 系统方法", methodTitle: "围绕策略建立显式执行闭环。",
    methodIntro: "只有执行层直接驱动机器人。外围 Harness 维护目标、状态、技能契约与完成证据。",
    flow1Title: "决策", flow1Text: "将当前观测和有效记忆组装为多模态 Agent 的决策上下文。",
    flow2Title: "规划", flow2Text: "生成有序子任务，并明确前置条件、完成谓词和动作预算。",
    flow3Title: "执行", flow3Text: "通过类型化接口调用已注册技能，由固定 π0 执行器生成动作。",
    flow4Title: "验证", flow4Text: "依据 Watchdog 反馈执行等待、结束、重放、恢复或重新规划。",
    memoryTitle: "动态记忆与事实有效性。",
    memoryText: "环境状态 Eₜ、任务状态 Tₜ、工具索引 Kₜ 与持久事实 Fₜ 构成记忆窗口。新观测优先于相矛盾的状态事实，只有有效事实能够进入下一次决策。",
    memoryCaption: "持久事实可以跨上下文重置保留；有效性检查防止过期状态进入 Agent 上下文。",
    watchdogTitle: "有界恢复与显式状态转移。",
    watchdogText: "Watchdog 按各技能配置的频率检查进度与完成证据。失败或停滞触发有界重试或恢复；预算耗尽后，对未完成后缀重新规划或报告失败。",
    watchdogCaption: "动作预算耗尽本身不代表成功，子任务完成必须有观测证据支持。",
    tasksLabel: "04 / 任务设置", tasksTitle: "规划、任务续接与扰动恢复。",
    task1Title: "连续整理", task1Text: "将两个积木分别移入目标容器并按下按钮，考察多步组合与子任务切换。",
    task2Title: "上下文中断", task2Text: "暂停 T1、清除 Agent 短期上下文，再请求继续执行，考察持久事实的作用。",
    task3Title: "扰动恢复", task3Text: "引入空抓、掉落、盒子移动或遮挡，考察重放、恢复和重新规划。",
    tasksCaption: "论文提供的任务协议示意图。物体与干预仅作说明，该图不是逐次试验记录。",
    resultsLabel: "05 / 实验结果", resultsTitle: "已观察任务成功率为 85%。", trialsMeta: "每种配置 20 次试验",
    resultsIntro: "所有配置均在实体 Piper 机械臂上使用 π0 作为底层执行器。成功要求完成整条指令，包括最后的按钮动作。",
    chartTitle: "端到端任务成功率", baseline: "单体 π0", decomposition: "π0 + 子任务分解", watchdogConfig: "π0 + Watchdog",
    gainTitle: "多完成 14 次成功试验。",
    gainText: "在当前评估设置中，完整 Harness 相比单体 π0 提高了 70 个百分点，相比仅添加任务分解提高了 20 个百分点。",
    gainCaveat: "这些数字是汇总任务完成率的观察差异，不是恢复效率、延迟或统计显著性的估计。",
    componentTitle: "独立加入一个组件后，有什么变化？",
    componentIntro: "每一行只向同一单体基线加入所标示的组件，各行是独立比较。",
    tableCaption: "V03 表 IV 的独立单组件比较，以及表 III 的完整系统参考结果",
    configuration: "配置", successes: "成功次数", successRate: "成功率", gainHeader: "相对 π0 的变化",
    shortTermConfig: "π0 + 短期记忆", factsConfig: "π0 + 跨会话事实", completeConfig: "HarnessVLA · 完整系统",
    tableNote: "来源：V03 表 III–IV。pp 表示百分点。每多一次成功试验，成功率增加 5 个百分点。完整系统与独立单组件配置分别评估。",
    scopeTitle: "证据支持的范围",
    scope1: "论文报告的是配置层面的汇总次数，没有单独列出 T1、T2 或 T3 的结果。因此不能据此确定各类扰动的恢复率，或上下文中断后的专项续接表现。",
    scope2: "独立添加组件不能测量组件交互，也不能说明从完整系统中移除某组件的影响。这些次数尚不能确定运行成本、跨机器人迁移能力或配对统计检验结果。",
    repoScope: "链接中的仓库是公开实现。当前 README 将 T3 恢复描述为在模拟执行器中验证；该实现范围应与论文任务定义及汇总实验结果区分理解。",
    resourcesLabel: "06 / 论文与代码", resourcesTitle: "阅读论文，查看实现。",
    resourcesIntro: "英文和中文 V03 论文均为 Word 格式的研究草稿。当前稿件使用匿名作者，尚未标注正式发表会议、期刊或 DOI。",
    englishPaper: "英文论文", chinesePaper: "中文论文", manuscriptVersion: "V03 · 研究草稿", sourceCode: "源代码",
    footer: "长时序机器人操作 · V03 论文展示页", backTop: "返回顶部 ↑", figureViewer: "论文图片", openOriginal: "打开原始图片 ↗"
  };
  let language = "en";
  let activeCaption = "";
  const toggle = document.getElementById("language-toggle");
  const dialog = document.getElementById("figure-dialog");
  const enlargedImage = document.getElementById("figure-dialog-image");
  const dialogCaption = document.getElementById("figure-dialog-caption");
  const originalLink = document.getElementById("figure-original-link");
  function applyLanguage(next) {
    language = next;
    const dictionary = language === "zh" ? chinese : english;
    document.documentElement.lang = language === "zh" ? "zh-Hans" : "en";
    document.title = language === "zh" ? "HarnessVLA | 长时序机器人操作" : "HarnessVLA | Long-Horizon Robot Manipulation";
    nodes.forEach(node => { node.textContent = dictionary[node.dataset.i18n] ?? english[node.dataset.i18n]; });
    toggle.innerHTML = language === "zh" ? 'English <span aria-hidden="true">↔</span>' : '中文 <span aria-hidden="true">↔</span>';
    toggle.setAttribute("aria-label", language === "zh" ? "Switch to English" : "切换为中文");
    document.getElementById("close-figure").setAttribute("aria-label", language === "zh" ? "关闭图片" : "Close figure");
    document.querySelectorAll("[data-figure]").forEach(button => {
      const alt = button.querySelector("img").alt;
      button.setAttribute("aria-label", language === "zh" ? `放大图片：${chinese[button.dataset.caption]}` : `Enlarge figure: ${alt}`);
    });
    if (activeCaption) dialogCaption.textContent = dictionary[activeCaption];
  }
  toggle.addEventListener("click", () => applyLanguage(language === "en" ? "zh" : "en"));
  document.querySelectorAll("[data-figure]").forEach(button => {
    button.addEventListener("click", () => {
      activeCaption = button.dataset.caption;
      enlargedImage.src = button.dataset.figure;
      enlargedImage.alt = button.querySelector("img").alt;
      originalLink.href = button.dataset.figure;
      dialogCaption.textContent = (language === "zh" ? chinese : english)[activeCaption];
      dialog.showModal();
    });
  });
  document.getElementById("close-figure").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", event => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });
  document.querySelector("#t1-video source").addEventListener("error", () => {
    document.getElementById("video-error").hidden = false;
  });
  document.getElementById("t1-video").addEventListener("error", () => {
    document.getElementById("video-error").hidden = false;
  });
  applyLanguage("en");
})();
