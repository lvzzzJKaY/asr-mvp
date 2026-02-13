# Android 打包（Capacitor）

## 前提

- 已安装 Node.js 18+
- 已安装 Android Studio

## 步骤

```bash
cd mobile
npm install
npm run android:add
npm run android:sync
npm run android:open
```

在 Android Studio 中运行到真机/模拟器。

## 关键配置

应用默认加载 `mobile/www/index.html`（由 `../web/index.html` 复制）。

上线后，打开 App 页面，将 `API Base URL` 设置为你的线上服务地址，例如：

- `https://your-app.onrender.com`

## 说明

- 录音权限由 Android 工程处理，首次会弹权限。
- 音色克隆需要服务端配置 `ELEVENLABS_API_KEY`。
