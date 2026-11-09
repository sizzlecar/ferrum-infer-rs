'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const active = new Map();
  const cards = [];
  let config = null;
  let mode = 'generate';
  let reference = null;
  let referenceVersion = 0;
  let loadingReference = false;
  let serial = 0;
  const PNG = [137, 80, 78, 71, 13, 10, 26, 10];
  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  function notice(message, success = false) {
    $('notice').textContent = message;
    $('notice').classList.toggle('success', success);
  }
  function update() {
    const limit = config ? Math.max(1, Math.min(2, config.maximum_concurrency || 2)) : 2;
    $('submit').disabled = !config || active.size >= limit || loadingReference;
    $('submit').textContent = !config ? '等待服务连接' : active.size >= limit ? '已有 2 个请求在等待' : mode === 'edit' ? '开始编辑 ↗' : '生成画面 ↗';
    $('capacity').textContent = active.size ? `正在等待 ${active.size}/${limit} 个请求 · 可逐项取消` : `最多同时等待 ${limit} 个请求`;
    $('seed').disabled = $('random').checked;
  }
  function setMode(next) {
    if (next === 'edit' && !config?.supports_edits) return;
    mode = next;
    for (const name of ['generate', 'edit']) {
      $(`mode-${name}`).classList.toggle('active', mode === name);
      $(`mode-${name}`).setAttribute('aria-pressed', String(mode === name));
    }
    $('reference-panel').hidden = mode !== 'edit';
    $('edit-hint').hidden = mode !== 'edit';
    $('prompt').placeholder = mode === 'edit' ? '希望修改什么？哪些内容需要保持不变？' : '描述主体、场景、光线和风格…';
    update();
  }
  function randomSeed() {
    const words = crypto.getRandomValues(new Uint32Array(2));
    return (words[0] & 0x1fffff) * 4294967296 + words[1];
  }
  function seedValue() {
    if ($('random').checked) return randomSeed();
    const raw = $('seed').value.trim();
    const value = Number(raw);
    if (!/^\d+$/.test(raw) || !Number.isSafeInteger(value) || value < 0) throw new Error('Seed 请输入 0–9007199254740991 之间的整数。');
    return value;
  }
  function setReference(blob, name) {
    if (reference) URL.revokeObjectURL(reference.url);
    reference = { blob, name, url: URL.createObjectURL(blob) };
    $('reference-preview').src = reference.url;
    $('reference-preview').hidden = false;
    $('reference-label').textContent = `${name} · ${(blob.size / 1048576).toFixed(1)} MB · 点击更换`;
  }
  async function selectFile(file) {
    if (!file) return;
    const version = ++referenceVersion;
    loadingReference = true;
    update();
    try {
      const signature = new Uint8Array(await file.slice(0, 8).arrayBuffer());
      if (PNG.every((byte, i) => signature[i] === byte)) {
        if (version === referenceVersion) setReference(file, file.name);
      } else if (signature[0] === 255 && signature[1] === 216 && signature[2] === 255) {
        const bitmap = await createImageBitmap(file);
        const canvas = document.createElement('canvas');
        try {
          canvas.width = bitmap.width;
          canvas.height = bitmap.height;
          const context = canvas.getContext('2d');
          if (!context) throw new Error('浏览器无法转换这张图片，请直接上传 PNG。');
          context.drawImage(bitmap, 0, 0);
          const png = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
          if (!png) throw new Error('JPG 转换失败，请直接上传 PNG。');
          if (version === referenceVersion) setReference(png, file.name.replace(/\.[^.]*$/, '') + '.png');
        } finally { bitmap.close(); canvas.width = 0; canvas.height = 0; }
      } else throw new Error('请选择 PNG 或 JPG 图片；实际上传格式必须为 PNG。');
      if (version === referenceVersion) notice('参考图已准备好。', true);
    } catch (error) { if (version === referenceVersion) notice(error.message || '读取图片失败。'); }
    finally { if (version === referenceVersion) { loadingReference = false; update(); } $('file').value = ''; }
  }
  async function copy(text) {
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(text);
      else {
        const input = element('textarea');
        input.value = text; input.style.position = 'fixed'; input.style.left = '-9999px';
        document.body.append(input); input.select();
        const copied = document.execCommand('copy'); input.remove();
        if (!copied) throw new Error('copy denied');
      }
      notice('已复制。', true);
    } catch (_) { notice('浏览器未允许复制，请手动选择对应文字。'); }
  }
  function action(parent, text, handler) {
    const button = element('button', 'small-button', text);
    button.type = 'button'; button.addEventListener('click', handler); parent.append(button);
    return button;
  }
  function prune() {
    const finished = cards.filter(card => !active.has(card.id));
    while (finished.length > 6) {
      const old = finished.pop();
      old.node.remove();
      if (old.url) URL.revokeObjectURL(old.url);
      cards.splice(cards.indexOf(old), 1);
    }
  }
  async function submit(event) {
    event.preventDefault();
    if (!config || $('submit').disabled) return;
    const prompt = $('prompt').value.trim();
    if (!prompt) { notice('先写下画面描述。'); $('prompt').focus(); return; }
    if (mode === 'edit' && !reference) { notice('请先选择一张参考图。'); return; }
    let seed;
    try { seed = seedValue(); } catch (error) { notice(error.message); return; }
    const request = { model: config.model, prompt, seed, size: $('size').value, n: 1, response_format: 'b64_json' };
    const editing = mode === 'edit';
    const original = reference?.blob;
    const controller = new AbortController();
    const id = ++serial;
    const node = element('article', 'card');
    const top = element('div', 'card-top');
    top.append(element('span', '', `${editing ? '图片编辑' : '文生图'} · ${id}`));
    const badge = element('span', 'badge', '等待返回'); top.append(badge);
    const preview = element('div', 'preview');
    const pending = element('div', 'pending-view');
    const elapsed = element('span', 'elapsed', '已等待 00:00');
    pending.append(element('span', 'spinner'), element('span', '', '正在等待服务返回图片'), elapsed);
    const cancel = action(pending, '取消等待', () => { controller.abort(); cancel.disabled = true; });
    cancel.classList.add('cancel'); preview.append(pending);
    const body = element('div', 'card-body');
    body.append(element('p', 'card-prompt', prompt), element('div', 'metadata', `${request.size} · Seed ${seed}`));
    const actions = element('div', 'actions');
    action(actions, '复制描述', () => copy(prompt)); action(actions, '复制 Seed', () => copy(String(seed)));
    body.append(actions); node.append(top, preview, body);
    const card = { id, node, url: null };
    cards.unshift(card); active.set(id, controller); $('history').prepend(node); $('empty').hidden = true;
    notice(''); update();
    const started = performance.now();
    const timer = setInterval(() => {
      const seconds = Math.floor((performance.now() - started) / 1000);
      elapsed.textContent = `已等待 ${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
    }, 1000);
    try {
      let response;
      if (editing) {
        const data = new FormData();
        for (const [key, value] of Object.entries(request)) data.append(key, String(value));
        data.append('image', original, 'reference.png');
        response = await fetch('/v1/images/edits', { method: 'POST', body: data, signal: controller.signal });
      } else response = await fetch('/v1/images/generations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request), signal: controller.signal });
      let payload;
      try { payload = await response.json(); } catch (_) { throw new Error(`服务返回了无法解析的响应（HTTP ${response.status}）。`); }
      if (!response.ok) throw new Error(payload?.error?.message || `请求失败（HTTP ${response.status}）。`);
      if (!Array.isArray(payload.data) || payload.data.length !== 1 || typeof payload.data[0].b64_json !== 'string') throw new Error('服务没有返回约定的单张 PNG 图片。');
      if (controller.signal.aborted) throw new DOMException('Aborted', 'AbortError');
      const binary = atob(payload.data[0].b64_json);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      if (!PNG.every((byte, i) => bytes[i] === byte)) throw new Error('服务返回的图片不是有效 PNG。');
      const blob = new Blob([bytes], { type: 'image/png' });
      card.url = URL.createObjectURL(blob);
      const image = element('img'); image.alt = prompt; image.src = card.url;
      await image.decode();
      if (controller.signal.aborted) throw new DOMException('Aborted', 'AbortError');
      preview.replaceChildren(image); badge.textContent = '已完成'; badge.classList.add('done');
      const download = element('a', 'small-button', '下载 PNG');
      download.href = card.url; download.download = `ferrum-${seed}-${request.size}.png`; actions.prepend(download);
      if (config.supports_edits) action(actions, '继续编辑', () => {
        ++referenceVersion; loadingReference = false; setReference(blob, `生成结果 · Seed ${seed}`); setMode('edit');
        $('prompt').value = ''; $('prompt').focus(); notice('已将这张结果设为参考图。', true);
        $('form').scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
      body.append(element('div', 'metadata', `本次等待 ${((performance.now() - started) / 1000).toFixed(1)} 秒`));
    } catch (error) {
      const cancelled = controller.signal.aborted || error.name === 'AbortError';
      badge.textContent = cancelled ? '已取消等待' : '请求失败'; badge.classList.add('error');
      const message = cancelled ? '已中止浏览器等待。服务端会按安全边界处理取消。' : error.message || '请求失败，请稍后重试。';
      preview.replaceChildren(element('div', 'pending-view error-copy', message));
      if (card.url) { URL.revokeObjectURL(card.url); card.url = null; }
    } finally { clearInterval(timer); active.delete(id); update(); prune(); }
  }
  $('form').addEventListener('submit', submit);
  $('mode-generate').addEventListener('click', () => setMode('generate'));
  $('mode-edit').addEventListener('click', () => setMode('edit'));
  $('random').addEventListener('change', update);
  $('drop-zone').addEventListener('click', () => $('file').click());
  $('file').addEventListener('change', () => selectFile($('file').files[0]));
  for (const name of ['dragenter', 'dragover']) $('drop-zone').addEventListener(name, event => { event.preventDefault(); $('drop-zone').classList.add('dragover'); });
  $('drop-zone').addEventListener('dragleave', () => $('drop-zone').classList.remove('dragover'));
  $('drop-zone').addEventListener('drop', event => { event.preventDefault(); $('drop-zone').classList.remove('dragover'); selectFile(event.dataTransfer.files[0]); });
  window.addEventListener('beforeunload', () => { for (const controller of active.values()) controller.abort(); });
  fetch('/playground/config', { cache: 'no-store' }).then(async response => {
    const value = await response.json();
    if (!response.ok) throw new Error(value?.error?.message || '图像服务暂不可用。');
    if (typeof value.model !== 'string' || !Array.isArray(value.sizes) || !value.sizes.length) throw new Error('服务配置缺少可用模型或尺寸。');
    config = value; $('model').textContent = config.model; $('model').title = config.model;
    $('size').replaceChildren();
    for (const size of config.sizes) {
      const option = element('option', '', size.label || `${size.width} × ${size.height}`);
      option.value = `${size.width}x${size.height}`; $('size').append(option);
    }
    const preferred = `${config.default_width}x${config.default_height}`;
    if ([...$('size').options].some(option => option.value === preferred)) $('size').value = preferred;
    $('size').disabled = false; $('mode-edit').disabled = !config.supports_edits;
    if (Number.isSafeInteger(config.default_seed) && config.default_seed >= 0) $('seed').value = String(config.default_seed);
    else notice('服务默认 Seed 超出浏览器安全范围，请使用随机 Seed 或输入安全范围内的整数。');
    update();
  }).catch(error => { config = null; $('model').textContent = '服务未就绪'; notice(error.message || '无法连接图像服务，请刷新页面重试。'); update(); });
})();
