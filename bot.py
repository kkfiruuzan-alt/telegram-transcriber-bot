import os
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from faster_whisper import WhisperModel

# Load env
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
MODEL_SIZE = os.getenv("MODEL_SIZE", "small")  
DEVICE = os.getenv("DEVICE", "cpu")            
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  
TARGET_LANG = os.getenv("TARGET_LANG", "he")   
MODE = os.getenv("MODE", "transcribe")         

model: Optional[WhisperModel] = None

def get_model() -> WhisperModel:
    global model
    if model is None:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return model

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "היי! שלחו לי הודעת קול/אודיו או וידאו, ואני אתמלל. "
        "פקודות: /mode, /lang, /help"
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/mode transcribe — תמלול לשפת המקור\n"
        "/mode translate — תרגום לאנגלית תוך כדי תמלול\n"
        "/lang he — לקבוע שפה מועדפת (he, en, ar, ...)\n"
        "שלחו הודעת קול (Voice) או קובץ אודיו/וידאו (MP3/OGG/WAV/MP4)"
    )

async def mode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global MODE
    if context.args and context.args[0] in {"transcribe", "translate"}:
        MODE = context.args[0]
        await update.message.reply_text(f"מצב עודכן ל־{MODE}")
    else:
        await update.message.reply_text("השתמשו: /mode transcribe | /mode translate")

async def lang_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TARGET_LANG
    if context.args:
        TARGET_LANG = context.args[0]
        await update.message.reply_text(f"שפה מועדפת עודכנה ל־{TARGET_LANG}")
    else:
        await update.message.reply_text("השתמשו: /lang he  (או השפה שתרצו)")

def ffmpeg_available() -> bool:
    from shutil import which
    return which("ffmpeg") is not None

async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    tg_file = None
    suggested_ext = "ogg"
    if msg.voice:
        tg_file = await msg.voice.get_file()
        suggested_ext = "ogg"
    elif msg.audio:
        tg_file = await msg.audio.get_file()
        suggested_ext = (msg.audio.file_name or "audio").split(".")[-1].lower() if msg.audio.file_name else "mp3"
    elif msg.video:
        tg_file = await msg.video.get_file()
        suggested_ext = "mp4"
    elif msg.document and msg.document.mime_type and "audio" in msg.document.mime_type:
        tg_file = await msg.document.get_file()
        suggested_ext = (msg.document.file_name or "audio").split(".")[-1].lower()
    else:
        await msg.reply_text("שלחו לי הודעת קול / קובץ אודיו / וידאו לתמלול.")
        return

    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / f"in.{suggested_ext}"
        out_wav = Path(td) / "out.wav"
        await tg_file.download_to_drive(in_path)

        if not ffmpeg_available():
            await msg.reply_text("FFmpeg לא מותקן על השרת.")
            return

        import subprocess
        cmd = [
            "ffmpeg", "-y", "-i", str(in_path),
            "-ac", "1", "-ar", "16000",
            str(out_wav)
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        if proc.returncode != 0 or not out_wav.exists():
            await msg.reply_text("שגיאה בהמרת אודיו.")
            return

        whisper = get_model()
        task = "translate" if MODE == "translate" else "transcribe"
        segments, info = whisper.transcribe(
            str(out_wav),
            language=(None if not TARGET_LANG else TARGET_LANG),
            task=task,
            beam_size=5,
            vad_filter=True,
        )
        text = "".join(seg.text for seg in segments).strip()
        if not text:
            text = "(לא זוהה טקסט)"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        await msg.reply_text(f"📝 {stamp}\n{text}")

def main():
    if not BOT_TOKEN:
        raise SystemExit("Missing BOT_TOKEN in environment. See .env.example")
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("mode", mode_cmd))
    app.add_handler(CommandHandler("lang", lang_cmd))

    app.add_handler(MessageHandler(
        filters.VOICE | filters.AUDIO | filters.VIDEO | (filters.Document.MimeType("audio/")), 
        handle_media
    ))

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
