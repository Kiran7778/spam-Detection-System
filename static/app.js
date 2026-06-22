document.addEventListener('DOMContentLoaded', () => {
    // --- State Variables ---
    let statsData = null;

    // --- DOM Elements ---
    const navItems = document.querySelectorAll('.nav-item');
    const tabs = document.querySelectorAll('.tab-content');
    const backendStatus = document.getElementById('backend-status');
    const backendStatusDot = document.querySelector('.status-dot');
    
    // Header Info
    const inferenceModeVal = document.getElementById('inference-mode-val');
    const awsModeVal = document.getElementById('aws-mode-val');
    const totalDatasetVal = document.getElementById('total-dataset-val');
    const queueCountBadge = document.getElementById('queue-count-badge');
    const queuePendingCount = document.getElementById('queue-pending-count');
    const statsThresholdVal = document.getElementById('stats-threshold-val');

    // Sandbox Tab Elements
    const sandboxForm = document.getElementById('sandbox-form');
    const messageInput = document.getElementById('message-input');
    const usernameInput = document.getElementById('username-input');
    const btnModerate = document.getElementById('btn-moderate');
    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultView = document.getElementById('result-view');
    const labelBadge = document.getElementById('label-badge');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceVal = document.getElementById('confidence-val');
    const elapsedVal = document.getElementById('elapsed-val');
    const flaggedAlert = document.getElementById('flagged-alert');
    const thresholdVal = document.getElementById('threshold-val');
    const presetTags = document.querySelectorAll('.preset-tag');

    // Review Tab Elements
    const btnRefreshQueue = document.getElementById('btn-refresh-queue');
    const reviewQueueTbody = document.getElementById('review-queue-tbody');

    // Analytics Tab Elements
    const trainSizeVal = document.getElementById('train-size-val');
    const valSizeVal = document.getElementById('val-size-val');
    const testSizeVal = document.getElementById('test-size-val');

    // --- Tab Navigation Switcher ---
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.getAttribute('data-tab');
            
            // Toggle active menu item
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            // Toggle active tab content
            tabs.forEach(tab => {
                if (tab.id === `tab-${targetTab}`) {
                    tab.classList.add('active');
                } else {
                    tab.classList.remove('active');
                }
            });

            // Perform specific load actions per tab
            if (targetTab === 'review') {
                loadReviewQueue();
            } else if (targetTab === 'analytics') {
                loadStats();
            }
        });
    });

    // --- Quick Presets ---
    presetTags.forEach(tag => {
        tag.addEventListener('click', () => {
            messageInput.value = tag.getAttribute('data-text');
            messageInput.focus();
        });
    });

    // --- Backend Health Check & Startup ---
    async function checkBackendHealth() {
        try {
            const res = await fetch('/health');
            const data = await res.json();
            
            if (res.ok && data.status === 'healthy') {
                backendStatus.textContent = 'API Server Connected';
                backendStatusDot.className = 'status-dot online';
                return true;
            } else {
                throw new Error('API reported unhealthy status');
            }
        } catch (err) {
            console.error('Backend connection failed:', err);
            backendStatus.textContent = 'Disconnected / Local Mode';
            backendStatusDot.className = 'status-dot offline';
            return false;
        }
    }

    // --- Load Stats Endpoint ---
    async function loadStats() {
        try {
            const res = await fetch('/api/stats');
            if (!res.ok) throw new Error('Failed to fetch stats');
            
            statsData = await res.json();
            
            // Render Stats across UI
            inferenceModeVal.textContent = statsData.inference_engine;
            awsModeVal.textContent = statsData.aws_mode;
            
            const totalSamples = statsData.dataset.train + statsData.dataset.val + statsData.dataset.test;
            totalDatasetVal.textContent = totalSamples.toLocaleString();
            
            // Badges and counts
            queueCountBadge.textContent = statsData.review_queue_pending;
            queuePendingCount.textContent = statsData.review_queue_pending;
            statsThresholdVal.textContent = statsData.confidence_threshold.toFixed(2);
            if (thresholdVal) thresholdVal.textContent = statsData.confidence_threshold.toFixed(2);

            // Analytics Tab Metrics
            trainSizeVal.textContent = statsData.dataset.train.toLocaleString() + ' messages';
            valSizeVal.textContent = statsData.dataset.val.toLocaleString() + ' messages';
            testSizeVal.textContent = statsData.dataset.test.toLocaleString() + ' messages';
            
        } catch (err) {
            console.error('Error loading stats:', err);
        }
    }

    // --- Sandbox Moderation Submission ---
    sandboxForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const messageText = messageInput.value.trim();
        const username = usernameInput.value.trim() || 'anonymous';
        
        if (!messageText) return;
        
        // Show loading state
        btnModerate.disabled = true;
        btnModerate.querySelector('span').textContent = 'Analyzing...';
        btnModerate.querySelector('i').className = 'fa-solid fa-spinner fa-spin';

        try {
            const res = await fetch('/moderate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: messageText, username: username })
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || 'Failed to moderate content');
            }

            const data = await res.json();
            renderSandboxResult(data);
            
            // Refresh stats to capture updated queue sizes if flagged
            loadStats();

        } catch (err) {
            alert(`Error: ${err.message}`);
        } finally {
            // Restore button state
            btnModerate.disabled = false;
            btnModerate.querySelector('span').textContent = 'Classify Message';
            btnModerate.querySelector('i').className = 'fa-solid fa-arrow-right';
        }
    });

    function renderSandboxResult(result) {
        // Toggle view container
        resultPlaceholder.classList.add('hidden');
        resultView.classList.remove('hidden');

        // Label update
        const label = result.label.toUpperCase();
        labelBadge.textContent = label;
        labelBadge.className = 'result-badge ' + result.label;

        // Confidence percentage and bar
        const confidencePct = (result.confidence * 100).toFixed(1);
        confidenceVal.textContent = `${confidencePct}%`;
        confidenceBar.style.width = `${confidencePct}%`;
        
        if (result.label === 'spam') {
            confidenceBar.style.background = 'var(--color-spam-gradient)';
        } else {
            confidenceBar.style.background = 'var(--color-ham-gradient)';
        }

        // Elapsed time
        elapsedVal.textContent = `${result.elapsed_ms.toFixed(2)} ms`;

        // Flagged alert banner
        if (result.flagged_for_review) {
            flaggedAlert.classList.remove('hidden');
        } else {
            flaggedAlert.classList.add('hidden');
        }
    }

    // --- Human Review Queue ---
    btnRefreshQueue.addEventListener('click', loadReviewQueue);

    async function loadReviewQueue() {
        reviewQueueTbody.innerHTML = `
            <tr class="table-placeholder">
                <td colspan="4" class="text-center">
                    <i class="fa-solid fa-spinner fa-spin table-placeholder-icon"></i>
                    <p>Fetching review queue...</p>
                </td>
            </tr>
        `;
        
        try {
            const res = await fetch('/api/review-queue');
            if (!res.ok) throw new Error('Failed to load review queue');
            
            const queueItems = await res.json();
            renderReviewQueue(queueItems);
            loadStats(); // Update badges too

        } catch (err) {
            console.error(err);
            reviewQueueTbody.innerHTML = `
                <tr class="table-placeholder">
                    <td colspan="4" class="text-center text-red">
                        <i class="fa-solid fa-circle-exclamation table-placeholder-icon"></i>
                        <p>Error loading queue: ${err.message}</p>
                    </td>
                </tr>
            `;
        }
    }

    function renderReviewQueue(items) {
        if (!items || items.length === 0) {
            reviewQueueTbody.innerHTML = `
                <tr class="table-placeholder">
                    <td colspan="4" class="text-center">
                        <i class="fa-solid fa-clipboard-check table-placeholder-icon"></i>
                        <p>No items pending human review. Excellent job!</p>
                    </td>
                </tr>
            `;
            return;
        }

        reviewQueueTbody.innerHTML = '';
        
        items.forEach((item, index) => {
            const dateStr = item.timestamp 
                ? new Date(item.timestamp * 1000).toLocaleString() 
                : 'N/A';
            
            const tr = document.createElement('tr');
            
            // Message text column with safety styling
            const tdMsg = document.createElement('td');
            const msgContainer = document.createElement('div');
            msgContainer.className = 'review-msg-text';
            msgContainer.textContent = item.text;
            tdMsg.appendChild(msgContainer);
            
            // Add metadata subtext if username exists
            const metaDiv = document.createElement('div');
            metaDiv.className = 'review-meta';
            metaDiv.innerHTML = `<span class="user"><i class="fa-regular fa-user"></i> ${item.username || 'anonymous'}</span>`;
            tdMsg.appendChild(metaDiv);
            
            // Date column
            const tdDate = document.createElement('td');
            tdDate.className = 'text-secondary';
            tdDate.textContent = dateStr;
            
            // Prediction column
            const tdPred = document.createElement('td');
            const confidencePct = (item.model_confidence * 100).toFixed(1);
            tdPred.innerHTML = `
                <span class="badge badge-outline ${item.model_prediction}">${item.model_prediction.toUpperCase()}</span>
                <span class="text-secondary" style="font-size: 12px; margin-left: 4px;">(${confidencePct}%)</span>
            `;
            
            // Action buttons column
            const tdAction = document.createElement('td');
            tdAction.className = 'action-cell';
            
            const btnGroup = document.createElement('div');
            btnGroup.className = 'action-buttons';
            
            const btnSpam = document.createElement('button');
            btnSpam.className = 'btn-action btn-spam-correct';
            btnSpam.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Spam';
            btnSpam.addEventListener('click', () => submitReviewCorrection(item, 'spam', tr));
            
            const btnHam = document.createElement('button');
            btnHam.className = 'btn-action btn-ham-correct';
            btnHam.innerHTML = '<i class="fa-solid fa-check"></i> Ham';
            btnHam.addEventListener('click', () => submitReviewCorrection(item, 'ham', tr));
            
            btnGroup.appendChild(btnHam);
            btnGroup.appendChild(btnSpam);
            tdAction.appendChild(btnGroup);
            
            // Assemble row
            tr.appendChild(tdDate);
            tr.appendChild(tdMsg);
            tr.appendChild(tdPred);
            tr.appendChild(tdAction);
            
            reviewQueueTbody.appendChild(tr);
        });
    }

    async function submitReviewCorrection(item, finalLabel, tableRow) {
        // Disable action buttons temporarily
        const buttons = tableRow.querySelectorAll('.btn-action');
        buttons.forEach(btn => btn.disabled = true);
        
        try {
            const res = await fetch('/api/submit-review', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: item.text,
                    prediction: item.model_prediction,
                    confidence: item.model_confidence,
                    corrected_label: finalLabel,
                    timestamp: item.timestamp
                })
            });

            if (!res.ok) throw new Error('Submission failed');

            // Visual removal feedback
            tableRow.style.transition = 'all 0.4s ease';
            tableRow.style.opacity = '0';
            tableRow.style.transform = 'translateX(50px)';
            
            setTimeout(() => {
                tableRow.remove();
                // Check if queue is now empty
                if (reviewQueueTbody.children.length === 0) {
                    reviewQueueTbody.innerHTML = `
                        <tr class="table-placeholder">
                            <td colspan="4" class="text-center">
                                <i class="fa-solid fa-clipboard-check table-placeholder-icon"></i>
                                <p>No items pending human review. Excellent job!</p>
                            </td>
                        </tr>
                    `;
                }
                loadStats(); // Update badge
            }, 400);

        } catch (err) {
            alert(`Failed to submit decision: ${err.message}`);
            buttons.forEach(btn => btn.disabled = false);
        }
    }

    // --- Init execution ---
    async function init() {
        const isUp = await checkBackendHealth();
        if (isUp) {
            loadStats();
        }
    }

    init();
});
