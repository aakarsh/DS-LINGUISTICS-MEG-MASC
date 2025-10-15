(defun sbatch-and-tail-remote (script-path)
  "Submit a Slurm script via TRAMP and tail its output.

This function is TRAMP-aware. If the SCRIPT-PATH is on a remote
machine (e.g., /ssh:user@host:/path/to/script), it runs 'sbatch'
on that host.

If it detects a multi-hop TRAMP path (e.g., involving a Docker
container like /ssh:host|docker:container:/path/), it intelligently
runs 'sbatch' on the base host, not inside the container."
  (interactive "fPath to sbatch script: ")
  (let* ((script-dir (file-name-directory script-path))
         ;; Determine the correct remote directory to run sbatch from.
         ;; This strips off any container layers (e.g., |docker:...).
         (remote-exec-dir (if (string-match "|" script-dir)
                              (car (split-string script-dir "|"))
                            script-dir))
         (output-buffer (generate-new-buffer "*sbatch-output*")))

    ;; Ensure we run the command in the correct remote directory
    (with-temp-buffer
      (let ((default-directory remote-exec-dir))
        (message "Running sbatch on host via TRAMP in %s" default-directory)

        ;; 1. Run sbatch remotely and capture its output
        (unless (zerop (call-process "sbatch" nil output-buffer nil script-path))
          (error "sbatch command failed. Check *sbatch-output* buffer."))

        ;; 2. Parse the output to get the job ID
        (with-current-buffer output-buffer
          (goto-char (point-min))
          (if (re-search-forward "Submitted batch job \\([0-9]+\\)" nil t)
              (let* ((job-id (match-string 1))
                     ;; Construct the log file path on the remote host
                     (log-file (expand-file-name (format "pytest-%s.out" job-id) remote-exec-dir))
                     (tail-command (format "tail -f %s" (shell-quote-argument log-file)))
                     (log-buffer-name (format "*slurm-log-%s*" job-id)))

                (message "Tailing log for job %s from %s" job-id log-file)
                ;; 3. Wait for the remote file system
                (sleep-for 0.5)

                ;; 4. Start tailing the remote log file. TRAMP handles the magic.
                (async-shell-command tail-command log-buffer-name)
                (switch-to-buffer-other-window log-buffer-name))
            (progn
              (display-buffer output-buffer)
              (error "Could not find job ID in sbatch output."))))))))


(defun sbatch-and-tail (script-path)
  "Submit a Slurm script with sbatch and tail its output file.

Prompts for the sbatch script to run. Assumes the sbatch output
is in the format 'Submitted batch job [JOB_ID]' and the log file
is named 'pytest-[JOB_ID].out' in the same directory."
  (interactive "fPath to sbatch script: ")
  (let ((output-buffer (generate-new-buffer "*sbatch-output*"))
        (default-directory (file-name-directory script-path)))
    ;; 1. Run sbatch and capture its output
    (unless (zerop (call-process "sbatch" nil output-buffer nil script-path))
      (error "sbatch command failed. Check *sbatch-output* buffer."))

    ;; 2. Parse the output to get the job ID
    (with-current-buffer output-buffer
      (goto-char (point-min))
      (if (search-forward-regexp "Submitted batch job \\([0-9]+\\)" nil t)
          (let* ((job-id (match-string 1))
                 (log-file (expand-file-name (format "pytest-%s.out" job-id)))
                 (tail-command (format "tail -f %s" (shell-quote-argument log-file)))
                 (log-buffer-name (format "*slurm-log-%s*" job-id)))
            (message "Tailing log for job %s in buffer %s..." job-id log-buffer-name)
            ;; 3. Wait a moment for the file system to create the log file
            (sleep-for 0.5)
            ;; 4. Start tailing the log file in a new buffer
            (async-shell-command tail-command log-buffer-name)
            (switch-to-buffer-other-window log-buffer-name))
        (progn
          (display-buffer output-buffer)
          (error "Could not find job ID in sbatch output."))))))

;; Optional: Bind this function to a convenient key, for example C-c s
(global-set-key (kbd "C-c s") 'sbatch-and-tail)


(provide 'custom)
