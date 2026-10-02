"use strict";
(() => {
  const nodes = Array.from(document.querySelectorAll("[data-i18n]"));
  const english = Object.fromEntries(nodes.map(node => [node.dataset.i18n, node.textContent]));
  const chinese = {
    "skip": "跳到正文",
    "navOverview": "概述",
    "navDemo": "演示",
    "navMethod": "方法",
    "navResults": "结果",
    "heroTag": "长时序机器人操作",
    "draftTag": "V04 · 研究草稿",
    "title": "面向有状态机器人操作的动态记忆与对象绑定技能框架",
    "heroDescription": "以固定 π0 为执行器，通过对象绑定技能调用、已验证任务状态与持久规则访问完成有状态操作。",
    "authors": "Zhang Haolin · V04 论文草稿",
    "paperButton": "论文下载 · 暂未开放",
    "codeButton": "代码",
    "demoButton": "观看演示",
    "stat1": "积木任务 · 17 / 20 次成功",
    "stat2": "A–B 转移 · 34 / 40 次成功",
    "stat3": "重置后 A–C 正确拒绝 · 19 / 20",
    "pp": "百分点",
    "enlarge": "查看原图 ↗",
    "architectureCaption": "V04 闭环连接决策、规划、执行与 Watchdog 监控，论文报告的实验使用固定 π0 执行器。",
    "overviewLabel": "01 / 研究概述",
    "overviewTitle": "让对象身份、任务状态与执行保持一致。",
    "abstract1": "长时程操作需要在连续动作生成之外维护任务状态、验证物理效果。HarnessVLA 在固定 π0 执行器外围协调任务分解、动态记忆、事实有效性检查与执行监控。",
    "abstract2": "Piper 真机实验包含积木与按钮操作，以及具有物质身份、槽位交换和预设禁止组合的试管液体转移。积木成功率为 17/20（85%）对 3/20（15%）；A–B 转移为 34/40（85%）对完整指令 π0 的 23/40（57.5%）。",
    "abstract3": "短期上下文重置后，保留持久规则访问时 A–C 正确拒绝为 19/20（95%），关闭访问时为 0/20。当前上下文含规则的参考条件为 20/20（100%）；三种规则条件的 A–B 错拒均为 0/20。",
    "demoLabel": "02 / 真机演示",
    "demoTitle": "Piper 上的积木与试管操作演示。",
    "demoDescription": "T1：将红色积木放入蓝色盒子，将绿色积木放入碗中，再按下红色按钮。",
    "videoFallback": "浏览器无法播放此视频，请使用下方下载链接。",
    "videoCaption": "T1 剪辑演示展示执行过程，成功率来自论文报告的试验次数。",
    "downloadVideo": "下载视频 ↓",
    "videoError": "视频加载失败，请尝试下载后播放。",
    "methodLabel": "03 / 系统方法",
    "methodTitle": "围绕策略建立显式执行闭环。",
    "methodIntro": "只有执行层直接驱动机器人。运行时绑定请求对象、检查预设约束，并依据物理效果证据推进任务。",
    "flow1Title": "决策",
    "flow1Text": "调用前读取持久知识、当前观测与任务状态。",
    "flow2Title": "规划",
    "flow2Text": "生成有序的对象绑定调用，明确前置条件和完成谓词。",
    "flow3Title": "执行",
    "flow3Text": "使用固定 π0 执行器调用已注册技能，每次调用具有独立 ID。",
    "flow4Title": "验证",
    "flow4Text": "验证本次调用的具体物理效果，再提交状态或推进任务。",
    "memoryTitle": "结合持久知识与最新场景状态。",
    "memoryText": "试管任务的 Mₜ = {L, Eₜ, Tₜ, K}：L 保存颜色映射和禁止组合，Eₜ 记录当前场景绑定，Tₜ 跟踪每支试管阶段，K 保存技能定义。新观测会使过期槽位记录失效。",
    "memoryCaption": "V04 图 2 同时展示试管与积木场景。持久身份与规则知识经有效性过滤后，与当前观测和任务状态共同进入决策上下文。",
    "watchdogTitle": "有界恢复与显式状态转移。",
    "watchdogText": "积木执行采用有界等待、结束、重放、恢复和后缀重规划。试管协议中，调用失败、证据不确定或预算耗尽会停止后续执行，不提交未经验证的效果。",
    "watchdogCaption": "V04 图 3 描述积木任务的 Watchdog；试管实验尚未验证自主试管恢复。",
    "tasksLabel": "04 / 积木与按钮任务",
    "tasksTitle": "积木基准及其任务条件。",
    "task1Title": "连续整理",
    "task1Text": "将两个积木分别移入目标容器并按下按钮，考察多步组合与子任务切换。",
    "task2Title": "上下文中断",
    "task2Text": "暂停 T1、清除 Agent 短期上下文，再请求继续执行，考察持久事实的作用。",
    "task3Title": "扰动恢复",
    "task3Text": "引入空抓、掉落、盒子移动或遮挡，考察重放、恢复和重新规划。",
    "tasksCaption": "论文提供的任务协议示意图。物体与干预仅作说明，该图不是逐次试验记录。",
    "resultsLabel": "05 / 积木实验结果",
    "resultsTitle": "积木与按钮：85% 对 15%。",
    "trialsMeta": "每种积木配置 20 次试验",
    "resultsIntro": "所有积木配置均在 Piper 真机上使用 π0 为底层执行器。成功要求完成整条指令，包括最后的按钮动作。",
    "chartTitle": "积木与按钮任务成功率",
    "baseline": "单体 π0",
    "decomposition": "π0 + 子任务分解",
    "watchdogConfig": "π0 + Watchdog",
    "gainTitle": "多完成 14 次成功试验。",
    "gainText": "在当前评估设置中，完整 Harness 相比单体 π0 提高了 70 个百分点，相比仅添加任务分解提高了 20 个百分点。",
    "gainCaveat": "这些数字是汇总任务完成率的观察差异，不是恢复效率、延迟或统计显著性的估计。",
    "componentTitle": "独立加入一个组件后，有什么变化？",
    "componentIntro": "每一行只向同一单体基线加入所标示的组件，各行是独立比较。",
    "tableCaption": "V04 表 IV 的积木独立组件比较及表 III 的完整系统结果",
    "configuration": "配置",
    "successes": "成功次数",
    "successRate": "成功率",
    "gainHeader": "相对 π0 的变化",
    "shortTermConfig": "π0 + 短期记忆",
    "factsConfig": "π0 + 跨会话事实",
    "completeConfig": "HarnessVLA · 完整系统",
    "tableNote": "来源：V04 表 III–IV。pp 表示百分点，每次积木试验成功对应 5 个百分点。完整系统另行评估。",
    "scopeTitle": "证据支持的范围",
    "scope1": "积木结果按配置汇总，没有独立的 T1–T3 次数。下方新增的试管规则记忆实验是另一项上下文重置测试。",
    "scope2": "独立添加组件不能测量组件交互，也不能说明从完整系统中移除某组件的影响。这些次数尚不能确定运行成本、跨机器人迁移能力或配对统计检验结果。",
    "repoScope": "图片与演示用于解释机制，汇总结果不包含逐次运行时间或已测量的故障分布。",
    "resourcesLabel": "07 / 论文与代码",
    "resourcesTitle": "了解方法，查看实现。",
    "resourcesIntro": "论文下载暂未开放，可通过下方链接查看公开实现。",
    "sourceCode": "源代码",
    "footer": "有状态机器人操作 · V04 论文展示页",
    "backTop": "返回顶部 ↑",
    "figureViewer": "论文图片",
    "openOriginal": "打开原始图片 ↗",
    "tubeDemoTitle": "交换槽位后的试管液体转移。",
    "tubeDemoIntro": "两段新视频演示将红色 A 与黄色 B 转入烧杯，并将用过的试管放入废物容器。",
    "g1DemoTitle": "G1 · A 在 1 号位，B 在 2 号位",
    "g2DemoTitle": "G2 · A 在 2 号位，B 在 1 号位",
    "g1DemoCaption": "红色 A 在 1 号槽位，黄色 B 在 2 号槽位。",
    "g2DemoCaption": "两种物质的试管交换了槽位。",
    "demoOrderNote": "两段演示先处理黄色 B，再处理红色 A；论文展示的是 A 先执行的计划示例。视频展示操作和槽位交换，试验统计另列于下方。",
    "bindingTitle": "区分物质身份、场景槽位与技能调用。",
    "bindingIntro": "交换槽位改变的是场景绑定，物质身份映射和已注册抓取技能保持稳定。",
    "bindingKnowledgeLabel": "持久知识",
    "bindingKnowledge": "A = 红 · B = 黄 · C = 蓝 · D = 绿",
    "bindingRule": "预设规则禁止 A–C 组合。",
    "bindingSceneLabel": "当前观测",
    "bindingScene": "G1：A → 1、B → 2 · G2：A → 2、B → 1",
    "bindingSceneText": "从当前场景读取槽位，使相矛盾的旧绑定失效。",
    "bindingCallLabel": "对象绑定调用",
    "bindingCallText": "独立调用 ID 将各支试管的完成证据分开，使重复技能调用互不混淆。",
    "tubeLabel": "06 / 试管实验",
    "tubeTitle": "转移指定物质，保留预设规则。",
    "tubeIntro": "第二项 Piper 任务使用四槽位试管架、烧杯和废物容器。A–B 请求允许执行，A–C 请求由明确提供的规则禁止。",
    "tubeCaption": "V04 图 5：允许的 A–B 任务包含两种交换槽位的摆放；A–C 面板展示依据预设禁止规则在执行前拒绝。",
    "protocolTitle": "六次调用，分别验证每支试管。",
    "protocolText": "每支指定试管依次抓取、向烧杯倾倒、弃管，再开始下一次抓取。注册表包含 grasp1–grasp4、pour 和 discard_tube。",
    "completionTitle": "完成判定要求物理效果。",
    "completionText": "两种指定液体均须进入烧杯，两支试管均须放入废物容器。所测终点不要求搅拌、定量体积或化学反应。",
    "tubeResultsTitle": "A–B 转移：85% 对 57.5%。",
    "tubeResultsIntro": "两种方法使用相同 π0 权重、相机视角、平台和完成标准。每种摆放下各方法测试 20 次，即每种方法 40 次，共 80 次。",
    "tubeHarnessStat": "HarnessVLA · 85%",
    "tubeBaselineStat": "完整指令 π0 · 57.5%",
    "tubeGainStat": "多完成 11 次成功试验",
    "tubeTableCaption": "V04 表 VII–VIII：两种摆放下的 A–B 液体转移结果",
    "arrangement": "摆放",
    "fullInstruction": "完整指令 π0",
    "g1Arrangement": "G1 · A 在 1，B 在 2",
    "g2Arrangement": "G2 · A 在 2，B 在 1",
    "combinedArrangements": "G1 + G2 合计",
    "tubeTableNote": "来源：V04 表 VII–VIII。该比较评估完整系统，不能分离单模块贡献、目标选择准确率或执行效率。",
    "ruleTitle": "上下文重置后，规则访问仍然重要。",
    "ruleIntro": "在 Piper 真机上，以相同 Agent 接口测试三种规则访问条件。每个条件分别包含 20 次 A–C 请求和 20 次允许的 A–B 请求。",
    "ruleTableCaption": "V04 表 IX：A–C 正确拒绝与 A–B 错误拒绝",
    "ruleCondition": "规则访问条件",
    "correctRefusal": "A–C 正确拒绝",
    "falseRefusal": "A–B 错误拒绝",
    "ruleCurrent": "当前上下文含规则",
    "rulePersistent": "重置后 · 保留持久规则访问",
    "ruleNoAccess": "重置后 · 关闭持久规则访问",
    "ruleTableNote": "来源：V04 表 IX。正确拒绝要求解释符合规则、执行器调用为零、无任务运动指令。仅静止或无关失败不算正确拒绝。",
    "ruleInterpretation": "重置后持久访问带来 95 个百分点的正确拒绝率差异。每个条件的 A–B 错拒均为零，说明观察到的 A–C 拒绝并非通过拒绝所有请求实现。",
    "tubeScopeTitle": "试管实验支持的证据范围",
    "tubeScope1": "A–B 成功衡量物理转移与弃管，A–C 拒绝衡量预设禁止规则的可用性。两者是独立终点，不能合并成一个成功率。",
    "tubeScope2": "所测槽位交换涉及 A、B 和 1–2 号位，尚不能确定四槽位操作、自主试管恢复、化学兼容性推断或跨机器人迁移表现。",
    "tubeScope3": "未正确拒绝本身不代表发生了禁止转移。规则由外部提供，并非自主学习；汇总次数也不能确定故障类别或运行成本。"
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
    document.title = language === "zh" ? "HarnessVLA | 有状态机器人操作" : "HarnessVLA | Stateful Robot Manipulation";
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
  document.querySelectorAll("video").forEach(video => {
    const error = document.querySelector(`[data-video-error="${video.id}"]`);
    const showError = () => { if (error) error.hidden = false; };
    video.addEventListener("error", showError);
    video.querySelector("source")?.addEventListener("error", showError);
    video.addEventListener("play", () => {
      document.querySelectorAll("video").forEach(other => { if (other !== video) other.pause(); });
    });
  });
  applyLanguage("en");
})();
