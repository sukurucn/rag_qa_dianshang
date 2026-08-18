<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { api, type ConversationTurn, type FeatureFlags, type IngestionJob, type QueryResult, type Session } from "./api";

type View = "chat" | "documents" | "faq" | "settings";

const view = ref<View>("chat");
const sessions = ref<Session[]>([]);
const activeSessionId = ref("");
const turns = ref<ConversationTurn[]>([]);
const question = ref("");
const sending = ref(false);
const errorMessage = ref("");
const lastResult = ref<QueryResult | null>(null);
const features = ref<FeatureFlags>({ faq_enabled: true, classifier_enabled: true });
const jobs = ref<IngestionJob[]>([]);
const documentFile = ref<File | null>(null);
const faqFile = ref<File | null>(null);
const statusMessage = ref("");

const activeSession = computed(() => sessions.value.find((item) => item.session_id === activeSessionId.value));

function showError(error: unknown) { errorMessage.value = error instanceof Error ? error.message : "操作失败"; }
async function loadSessions() {
  sessions.value = await api.listSessions();
  if (!activeSessionId.value && sessions.value[0]) await selectSession(sessions.value[0].session_id);
}
async function selectSession(sessionId: string) {
  activeSessionId.value = sessionId;
  turns.value = await api.getTurns(sessionId);
  lastResult.value = null;
}
async function createSession() {
  try {
    const session = await api.createSession(activeSession.value?.memory_turn_limit ?? 12);
    sessions.value.unshift(session); await selectSession(session.session_id); view.value = "chat";
  } catch (error) { showError(error); }
}
async function sendQuestion() {
  if (!question.value.trim() || sending.value) return;
  try {
    errorMessage.value = "";
    if (!activeSessionId.value) await createSession();
    sending.value = true;
    lastResult.value = await api.query(question.value.trim(), activeSessionId.value);
    question.value = "";
    turns.value = await api.getTurns(activeSessionId.value);
    await loadSessions();
  } catch (error) { showError(error); } finally { sending.value = false; }
}
async function updateMemoryLimit(event: Event) {
  if (!activeSession.value) return;
  try {
    const memory_turn_limit = Number((event.target as HTMLInputElement).value);
    const updated = await api.updateSession(activeSession.value.session_id, { memory_turn_limit });
    sessions.value = sessions.value.map((item) => item.session_id === updated.session_id ? updated : item);
  } catch (error) { showError(error); }
}
async function updateFeature(key: keyof FeatureFlags, value: boolean) {
  try { features.value = await api.updateFeatures({ [key]: value }); } catch (error) { showError(error); }
}
async function refreshJobs() { try { jobs.value = await api.listJobs(); } catch (error) { showError(error); } }
async function uploadDocument() {
  if (!documentFile.value) return;
  try { statusMessage.value = "文档已提交，正在处理中。"; await api.uploadDocument(documentFile.value); documentFile.value = null; await refreshJobs(); } catch (error) { showError(error); }
}
async function importFaq() {
  if (!faqFile.value) return;
  try {
    const result = await api.importFaq(faqFile.value);
    statusMessage.value = `FAQ 导入完成：新增 ${result.created_count}，更新 ${result.updated_count}，Redis 预热 ${result.redis_warmed_count}。`;
    faqFile.value = null;
  } catch (error) { showError(error); }
}
async function removeSession() {
  if (!activeSessionId.value) return;
  try { await api.deleteSession(activeSessionId.value); activeSessionId.value = ""; turns.value = []; await loadSessions(); } catch (error) { showError(error); }
}

onMounted(async () => {
  try { await Promise.all([loadSessions(), refreshJobs().catch(() => undefined), api.getFeatures().then((value) => { features.value = value; })]); }
  catch (error) { showError(error); }
});
</script>

<template>
  <main class="workbench">
    <aside class="sidebar">
      <div class="brand"><strong>RAG Agentic</strong><span>本地知识库工作台</span></div>
      <button class="primary" type="button" @click="createSession">新建会话</button>
      <nav aria-label="功能导航">
        <button :class="{ active: view === 'chat' }" @click="view = 'chat'">会话问答</button>
        <button :class="{ active: view === 'documents' }" @click="view = 'documents'">文档更新</button>
        <button :class="{ active: view === 'faq' }" @click="view = 'faq'">FAQ 更新</button>
        <button :class="{ active: view === 'settings' }" @click="view = 'settings'">响应设置</button>
      </nav>
      <section class="side-section">
        <h2>会话列表</h2>
        <button v-for="session in sessions" :key="session.session_id" class="session-item" :class="{ selected: session.session_id === activeSessionId }" @click="selectSession(session.session_id)">
          {{ session.title }}
        </button>
      </section>
      <section class="side-section controls">
        <h2>响应模块</h2>
        <label><input type="checkbox" :checked="features.faq_enabled" @change="updateFeature('faq_enabled', ($event.target as HTMLInputElement).checked)" /> FAQ（MySQL + Redis）</label>
        <label><input type="checkbox" :checked="features.classifier_enabled" @change="updateFeature('classifier_enabled', ($event.target as HTMLInputElement).checked)" /> 意图分类</label>
      </section>
    </aside>

    <section class="content">
      <p v-if="errorMessage" class="notice error">{{ errorMessage }}</p>
      <p v-if="statusMessage" class="notice">{{ statusMessage }}</p>
      <template v-if="view === 'chat'">
        <header class="page-header"><div><p class="eyebrow">会话问答</p><h1>{{ activeSession?.title ?? '新会话' }}</h1></div><button class="secondary" type="button" @click="removeSession" :disabled="!activeSessionId">删除会话</button></header>
        <div class="memory-bar"><label>历史保留轮数 <input type="range" min="0" max="256" :value="activeSession?.memory_turn_limit ?? 12" @change="updateMemoryLimit" /></label><strong>{{ activeSession?.memory_turn_limit ?? 12 }} 轮</strong><span>系统仅选择相关历史参与本次回答。</span></div>
        <div class="conversation" aria-live="polite">
          <article v-for="turn in turns" :key="turn.turn_number" class="turn"><div class="question"><span>用户</span><p>{{ turn.user_question }}</p></div><div class="answer"><span>助手</span><p>{{ turn.assistant_answer }}</p></div></article>
          <div v-if="!turns.length" class="empty">开始一个新会话。历史记忆会按本侧栏设置保留。</div>
        </div>
        <div v-if="lastResult" class="answer-meta">本次来源：{{ lastResult.source === 'faq' ? 'FAQ' : 'RAG' }}；使用历史 {{ lastResult.selected_memory_turns }} 轮；{{ lastResult.web_search_used ? '已使用网络补充。' : '未使用网络补充。' }}</div>
        <form class="composer" @submit.prevent="sendQuestion"><textarea v-model="question" placeholder="输入问题，按发送提交" :disabled="sending" /><button class="primary" type="submit" :disabled="sending || !question.trim()">{{ sending ? '处理中' : '发送' }}</button></form>
      </template>

      <template v-else-if="view === 'documents'">
        <header class="page-header"><div><p class="eyebrow">知识库更新</p><h1>文档入库</h1></div><button class="secondary" @click="refreshJobs">刷新任务</button></header>
        <section class="panel"><h2>上传文档</h2><p>支持 TXT、MD、DOCX、PDF、PPT、PPTX，单文件不超过 100MB。</p><input type="file" accept=".txt,.md,.docx,.pdf,.ppt,.pptx" @change="documentFile = ($event.target as HTMLInputElement).files?.[0] ?? null" /><button class="primary" :disabled="!documentFile" @click="uploadDocument">提交入库</button></section>
        <section class="panel"><h2>入库任务</h2><table><thead><tr><th>文件</th><th>状态</th><th>块数量</th><th>更新时间</th></tr></thead><tbody><tr v-for="job in jobs" :key="job.job_id"><td>{{ job.original_filename }}</td><td>{{ job.status }}</td><td>{{ job.stored_chunks }}</td><td>{{ new Date(job.updated_at).toLocaleString() }}</td></tr></tbody></table></section>
      </template>

      <template v-else-if="view === 'faq'">
        <header class="page-header"><div><p class="eyebrow">知识库更新</p><h1>FAQ 导入</h1></div></header>
        <section class="panel"><h2>上传 CSV 或 JSONL</h2><p>固定字段：<code>question,answer</code>。导入完成后会原子预热 Redis 问题缓存。</p><input type="file" accept=".csv,.jsonl" @change="faqFile = ($event.target as HTMLInputElement).files?.[0] ?? null" /><button class="primary" :disabled="!faqFile" @click="importFaq">导入 FAQ</button></section>
      </template>

      <template v-else>
        <header class="page-header"><div><p class="eyebrow">响应设置</p><h1>功能与记忆设置</h1></div></header>
        <section class="panel setting-list"><label>FAQ 匹配 <input type="checkbox" :checked="features.faq_enabled" @change="updateFeature('faq_enabled', ($event.target as HTMLInputElement).checked)" /></label><label>意图分类 <input type="checkbox" :checked="features.classifier_enabled" @change="updateFeature('classifier_enabled', ($event.target as HTMLInputElement).checked)" /></label><p>关闭意图分类后，当前问题直接进入 Milvus 检索。联网搜索始终由回答 Agent 在本地检索后自行决定。</p></section>
      </template>
    </section>
  </main>
</template>
