from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
import subprocess
import sys

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'video')
app.config['RESULTS_FOLDER'] = os.path.join(os.getcwd(), 'tracking_results')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5 GB limit
app.secret_key = 'change-me-for-production'

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'hevc', 'h265'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('沒有上傳檔案')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('未選擇檔案')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            flash(f'上傳成功: {filename}')
            return redirect(url_for('index'))
        else:
            flash('不支援的檔案類型')
            return redirect(request.url)

    # GET: 列出目前 video/ 以及 tracking_results/ 中的檔案
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    files = sorted([f for f in os.listdir(app.config['UPLOAD_FOLDER']) if not f.startswith('.')])
    results = sorted([f for f in os.listdir(app.config['RESULTS_FOLDER']) if not f.startswith('.')])
    return render_template('index.html', files=files, results=results)

@app.route('/video/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/results/<path:filename>')
def download_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, as_attachment=True)


@app.route('/run_detection', methods=['POST'])
def run_detection():
    """在背景啟動 检测人流.py 並將輸出寫入 tracking_results/detection_run.log"""
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    log_path = os.path.join(app.config['RESULTS_FOLDER'], 'detection_run.log')
    python_exec = sys.executable
    script_path = os.path.join(os.getcwd(), '检测人流.py')

    try:
        # 以背景程序啟動，將 stdout/stderr 附加到 log
        log_file = open(log_path, 'ab')
        process = subprocess.Popen([python_exec, script_path], stdout=log_file, stderr=log_file)
        flash(f'已啟動檢測（PID {process.pid}），請稍候查看 tracking_results/detection_run.log')
    except Exception as e:
        flash(f'啟動檢測失敗: {e}')

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
