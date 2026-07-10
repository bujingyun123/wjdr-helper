"""坐标拾取工具：打开截图，鼠标左键点击任意位置即打印 (x, y) 坐标。
用法：
    python tools/pick_coords.py                                      # 默认打开 screen.png
    python tools/pick_coords.py d:\workspace\wujindongri\shot.png    # 指定任意路径的截图
    python tools/pick_coords.py Screenshot_20260226.png              # 指定文件名
    python tools/pick_coords.py --capture                            # 自动截图后打开
    python tools/pick_coords.py --capture --out my_shot.png          # 截图并保存为指定文件名
    python tools/pick_coords.py --capture --device 127.0.0.1:5565   # 指定模拟器设备

参数说明：
    image           截图文件路径（位置参数，默认 screen.png）
    --capture       执行 ADB 截图后再打开
    --out PATH      --capture 时截图保存的文件名（默认与 image 参数一致）
    --adb PATH      adb 可执行文件路径
    --device ID     ADB 设备 ID（默认 127.0.0.1:5555）
"""
from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def capture_screenshot(
    adb: str = r"D:\leidian\LDPlayer9\adb.exe",
    device: str = "127.0.0.1:5555",
    out: Path = Path("screen.png"),
) -> bool:
    """用 adb pull 方式安全截图（避免 PowerShell 重定向损坏二进制）。"""
    print(f"[截图] 正在截图 ({device})…")
    try:
        subprocess.run(
            [adb, "-s", device, "shell", "screencap", "-p", "/sdcard/_bot_screen.png"],
            check=True, capture_output=True,
        )
        subprocess.run(
            [adb, "-s", device, "pull", "/sdcard/_bot_screen.png", str(out)],
            check=True, capture_output=True,
        )
        print(f"[截图] 已保存到 {out.resolve()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[截图失败] {e.stderr.decode(errors='ignore').strip() or e}")
        return False
    except FileNotFoundError:
        print(f"[截图失败] 找不到 adb 可执行文件: {adb}")
        return False


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="坐标拾取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "image", nargs="?", default="screen.png",
        help="截图文件路径（默认: screen.png）",
    )
    parser.add_argument("--capture", action="store_true", help="自动截图后打开")
    parser.add_argument(
        "--out", default=None, metavar="PATH",
        help="--capture 时截图保存路径（默认与 image 参数一致）",
    )
    parser.add_argument(
        "--adb", default=r"D:\leidian\LDPlayer9\adb.exe",
        metavar="PATH", help="adb 可执行文件路径",
    )
    parser.add_argument(
        "--device", default="127.0.0.1:5555",
        metavar="ID", help="ADB 设备 ID（默认: 127.0.0.1:5555）",
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    # --out 独立控制截图保存位置；未指定时与 image 同路径
    save_path = Path(args.out) if args.out else img_path

    if args.capture:
        if not capture_screenshot(adb=args.adb, device=args.device, out=save_path):
            print("截图失败，请先执行: adb connect 127.0.0.1:5555")
            return
        # 若 --out 与 image 不同，把 img_path 指向刚截的文件
        img_path = save_path

    if not img_path.exists():
        print(f"[错误] 找不到截图文件: {img_path.resolve()}")
        print("提示：")
        print("  1. 直接传文件名:  python tools/pick_coords.py 你的截图.png")
        print("  2. 自动截图:      python tools/pick_coords.py --capture --out 你的截图.png")
        return

    root = tk.Tk()
    root.title(f"坐标拾取 — {img_path.resolve()}  （左键=记录  右键=复制  导出=保存列表）")

    coords_log: list[tuple[int, int]] = []

    if HAS_PIL:
        img = Image.open(img_path)
        orig_w, orig_h = img.size

        # 按屏幕高度自动缩放（最大 900px 高）
        screen_h = root.winfo_screenheight()
        max_h = int(screen_h * 0.85)
        scale = min(1.0, max_h / orig_h)
        display_w = int(orig_w * scale)
        display_h = int(orig_h * scale)

        img_resized = img.resize((display_w, display_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img_resized)
    else:
        # 无 Pillow：用 Tkinter 原生加载（仅支持 PNG）
        photo = tk.PhotoImage(file=str(img_path))
        orig_w, orig_h = photo.width(), photo.height()
        display_w, display_h = orig_w, orig_h
        scale = 1.0
        print("[提示] 未安装 Pillow，图片不会自动缩放。pip install pillow 可启用缩放功能。")

    canvas = tk.Canvas(root, width=display_w, height=display_h, cursor="crosshair")
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=photo)

    # 状态栏
    status_var = tk.StringVar(value="鼠标左键点击截图以获取坐标")
    status_bar = tk.Label(root, textvariable=status_var, bd=1, relief="sunken", anchor="w",
                          font=("Consolas", 10), bg="#222", fg="#9cdcfe")
    status_bar.pack(fill="x")

    # 输出日志框
    log_frame = tk.Frame(root)
    log_frame.pack(fill="both", expand=True, padx=4, pady=4)

    log_text = tk.Text(log_frame, height=8, font=("Consolas", 10),
                       bg="#1e1e1e", fg="#d4d4d4", state="disabled")
    scrollbar = tk.Scrollbar(log_frame, command=log_text.yview)
    log_text.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    log_text.pack(side="left", fill="both", expand=True)

    def _log(msg: str) -> None:
        log_text.configure(state="normal")
        log_text.insert("end", msg + "\n")
        log_text.see("end")
        log_text.configure(state="disabled")
        print(msg)

    def on_click(event: tk.Event) -> None:
        # 将显示坐标换算回原始分辨率坐标
        real_x = int(event.x / scale)
        real_y = int(event.y / scale)
        coords_log.append((real_x, real_y))
        msg = f"({real_x:4d}, {real_y:4d})    # 显示坐标 ({event.x}, {event.y})"
        status_var.set(f"已记录: ({real_x}, {real_y})")
        _log(msg)

        # 在画布上画一个红点标记
        r = 4
        canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r,
                            fill="red", outline="yellow", width=1)
        canvas.create_text(event.x + 8, event.y - 8,
                           text=f"{real_x},{real_y}", fill="yellow",
                           font=("Consolas", 8))

    def on_right_click(_event: tk.Event) -> None:
        if not coords_log:
            return
        last = coords_log[-1]
        root.clipboard_clear()
        root.clipboard_append(f"({last[0]}, {last[1]})")
        status_var.set(f"已复制: ({last[0]}, {last[1]})")

    def on_export() -> None:
        if not coords_log:
            messagebox.showinfo("提示", "还没有记录任何坐标")
            return
        lines = [f"    ({x}, {y}, '描述')," for x, y in coords_log]
        result = "[\n" + "\n".join(lines) + "\n]"
        # 导出文件与截图同目录，方便对照
        out_path = img_path.with_suffix(".coords.txt")
        Path("tools").mkdir(exist_ok=True)
        out_path.write_text(result, encoding="utf-8")
        messagebox.showinfo("导出成功", f"已保存 {len(coords_log)} 个坐标到:\n{out_path.resolve()}")
        print(f"\n=== 导出坐标列表 ===\n{result}")

    def on_clear() -> None:
        coords_log.clear()
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo)
        log_text.configure(state="normal")
        log_text.delete("1.0", "end")
        log_text.configure(state="disabled")
        status_var.set("已清空，重新点击截图以获取坐标")

    canvas.bind("<Button-1>", on_click)
    canvas.bind("<Button-3>", on_right_click)

    # 工具栏按钮
    btn_frame = tk.Frame(root, bg="#2d2d2d")
    btn_frame.pack(fill="x", padx=4, pady=2)
    tk.Button(btn_frame, text="导出坐标列表", command=on_export,
              bg="#0e639c", fg="white", relief="flat", padx=8).pack(side="left", padx=4)
    tk.Button(btn_frame, text="清空标记", command=on_clear,
              bg="#5a3030", fg="white", relief="flat", padx=8).pack(side="left", padx=4)
    tk.Label(btn_frame, text="左键=记录  右键=复制最后坐标  坐标已换算为原始分辨率",
             bg="#2d2d2d", fg="#888", font=("", 9)).pack(side="right", padx=8)

    root.mainloop()


if __name__ == "__main__":
    main()
