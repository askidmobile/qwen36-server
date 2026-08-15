# WebUI Accessibility & Attachment Validation Script

## 1. Keyboard Navigation Flow
- **Tab from Header to Chat:**
  1. API Key input (`aria-label="API Key"`).
  2. Mode selector (`aria-label="Режим генерации"`).
  3. Model selector (`aria-label="Выбор модели"`).
  4. Apply button (`title="Переключить модель/контекст"`).
  5. New chat button (`aria-label="Новый чат"`).
  6. Attachment button (`aria-label="Прикрепить медиафайл"`).
  7. Composer textarea (`aria-label="Текст сообщения"`).
  8. Send button (`Отправить`).

## 2. Attachment Flow & Screen Reader Announcements
1. **Adding file:**
   - Press Enter/Space on `[📎]` button.
   - Select image or video from native dialog.
   - Screen reader announces: `"Прикреплено файлов: 1"`.
   - Attachment card renders in tray with thumbnail and `"Удалить <filename>"` button.
2. **Deleting file:**
   - Tab to `✕` button on card, press Enter.
   - Screen reader announces: `"Файл <filename> удалён"`.
   - Focus returns deterministically to `[📎]` button.
3. **Stateless Reload Check:**
   - On page refresh, messages in history display text placeholders `[Изображение]` or `[Видео]`.
   - No raw bytes or blob URLs stored in `localStorage`.
