/**
 * Watch Finder Application
 * Modern JavaScript with ES6+ features, modular design, and accessibility
 */

// Configuration
const CONFIG = {
    API_BASE_URL: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
        ? 'http://localhost:5001/api' 
        : 'https://web-production-a75cb.up.railway.app/api',
    SWIPE_THRESHOLD: 50,
    DRAG_THRESHOLD: 100,
    ANIMATION_DURATION: 300,
    MAX_RETRIES: 3,
    RETRY_DELAY: 1000
};

// Application State Manager
class AppState {
    constructor() {
        this.currentWatches = [];
        this.currentIndex = 0;
        this.likedWatches = [];
        this.dislikedWatches = [];
        this.step = 0;
        this.sessionId = null;
        this.isLoading = false;
        this.currentView = 'discover';
        
        // Touch/drag state
        this.dragState = {
            isDragging: false,
            startX: 0,
            startY: 0,
            currentX: 0,
            currentY: 0
        };
    }

    // State getters
    get currentWatch() {
        return this.currentWatches[this.currentIndex] || null;
    }

    get totalLiked() {
        return this.likedWatches.length;
    }

    get totalHistory() {
        return this.likedWatches.length + this.dislikedWatches.length;
    }

    // State mutations
    addLikedWatch(watch) {
        if (!this.likedWatches.some(w => w.index === watch.index)) {
            this.likedWatches.push(watch);
            this.removePreviousDislike(watch.index);
            this.updateLikedCount();
        }
    }

    addDislikedWatch(watch) {
        if (!this.dislikedWatches.some(w => w.index === watch.index)) {
            this.dislikedWatches.push(watch);
            this.removePreviousLike(watch.index);
        }
    }

    removePreviousLike(watchIndex) {
        this.likedWatches = this.likedWatches.filter(w => w.index !== watchIndex);
        this.updateLikedCount();
    }

    removePreviousDislike(watchIndex) {
        this.dislikedWatches = this.dislikedWatches.filter(w => w.index !== watchIndex);
    }

    updateLikedCount() {
        const countElement = document.getElementById('liked-count');
        if (countElement) {
            countElement.textContent = this.totalLiked;
        }
    }

    reset() {
        this.currentWatches = [];
        this.currentIndex = 0;
        this.step = 0;
        this.sessionId = null;
        this.updateLikedCount();
    }
}

// API Service
class ApiService {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
            ...options
        };

        try {
            const response = await fetch(url, defaultOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.status === 'error') {
                throw new Error(data.message || 'API request failed');
            }
            
            return data;
        } catch (error) {
            console.error(`API Error [${endpoint}]:`, error);
            throw error;
        }
    }

    async checkHealth() {
        const data = await this.request('/health');
        
        if (!data.engine_initialized) {
            throw new Error('Recommendation engine not initialized');
        }
        
        return data;
    }

    async startSession(numSeeds = 1) {
        return this.request('/start-session', {
            method: 'POST',
            body: JSON.stringify({ num_seeds: numSeeds })
        });
    }

    async getRecommendations(likedIndices, dislikedIndices, currentCandidates, step) {
        return this.request('/get-recommendations', {
            method: 'POST',
            body: JSON.stringify({
                liked_indices: likedIndices,
                disliked_indices: dislikedIndices,
                current_candidates: currentCandidates,
                step: step
            })
        });
    }

    async getVariants(watchIndex) {
        return this.request(`/get-variants?index=${watchIndex}`);
    }
}

// UI Manager
class UIManager {
    constructor() {
        this.screens = new Map();
        this.views = new Map();
        this.currentScreen = null;
        this.currentView = null;
        
        this.initializeScreens();
        this.initializeViews();
    }

    initializeScreens() {
        const screenElements = document.querySelectorAll('.screen');
        screenElements.forEach(screen => {
            this.screens.set(screen.id, screen);
        });
    }

    initializeViews() {
        const viewElements = document.querySelectorAll('.view');
        viewElements.forEach(view => {
            this.views.set(view.id, view);
        });
    }

    showScreen(screenId) {
        // Hide all screens
        this.screens.forEach(screen => {
            screen.classList.add('hidden');
        });

        // Show target screen
        const targetScreen = this.screens.get(screenId);
        if (targetScreen) {
            targetScreen.classList.remove('hidden');
            this.currentScreen = screenId;
            
            // Announce screen change for screen readers
            this.announceScreenChange(screenId);
        }
    }

    showView(viewId) {
        // Hide all views
        this.views.forEach(view => {
            view.classList.remove('active');
        });

        // Show target view
        const targetView = this.views.get(viewId);
        if (targetView) {
            targetView.classList.add('active');
            this.currentView = viewId;
            
            // Update navigation
            this.updateNavigation(viewId);
        }
    }

    updateNavigation(activeView) {
        const navButtons = document.querySelectorAll('.nav-btn');
        navButtons.forEach(btn => {
            btn.classList.remove('active');
        });

        // Map view IDs to button functions
        const viewMap = {
            'discover-view': 'showDiscover',
            'liked-view': 'showLiked',
            'history-view': 'showHistory'
        };

        const buttonFunction = viewMap[activeView];
        if (buttonFunction) {
            const activeButton = document.querySelector(`[onclick*="${buttonFunction}"]`);
            if (activeButton) {
                activeButton.classList.add('active');
            }
        }
    }

    showError(message) {
        const errorMessage = document.getElementById('error-message');
        if (errorMessage) {
            errorMessage.textContent = message;
        }
        this.showScreen('error-screen');
    }

    announceScreenChange(screenId) {
        const announcements = {
            'loading-screen': 'Loading application',
            'welcome-screen': 'Welcome screen',
            'main-app': 'Main application',
            'error-screen': 'Error occurred'
        };

        const announcement = announcements[screenId];
        if (announcement) {
            this.announce(announcement);
        }
    }

    announce(message) {
        // Create or update live region for screen reader announcements
        let liveRegion = document.getElementById('live-region');
        if (!liveRegion) {
            liveRegion = document.createElement('div');
            liveRegion.id = 'live-region';
            liveRegion.setAttribute('aria-live', 'polite');
            liveRegion.setAttribute('aria-atomic', 'true');
            liveRegion.style.position = 'absolute';
            liveRegion.style.left = '-10000px';
            liveRegion.style.width = '1px';
            liveRegion.style.height = '1px';
            liveRegion.style.overflow = 'hidden';
            document.body.appendChild(liveRegion);
        }
        
        liveRegion.textContent = message;
    }
}

// Watch Card Manager
class WatchCardManager {
    constructor(container) {
        this.container = container;
    }

    createWatchCard(watch) {
        if (!watch) return null;

        const card = document.createElement('div');
        card.className = 'watch-card';
        card.dataset.watchIndex = watch.index;
        card.setAttribute('role', 'article');
        card.setAttribute('aria-label', `Watch: ${watch.brand} ${watch.model_name}`);

        // Watch data with fallbacks
        const brand = watch.brand || 'Unknown Brand';
        const modelName = watch.model_name || 'Unknown Model';
        const price = watch.price || 'Price not available';
        const imageUrl = watch.image_url || this.getPlaceholderImage();
        const description = watch.description || '';

        // Specifications
        const specs = this.buildSpecifications(watch);
        const variantInfo = this.buildVariantInfo(watch);
        const descriptionHtml = this.buildDescription(description);

        card.innerHTML = `
            <img src="${imageUrl}" 
                 alt="${brand} ${modelName}" 
                 class="watch-image"
                 onerror="this.src='${this.getPlaceholderImage()}'">
            <div class="watch-info">
                <div class="watch-brand">${this.escapeHtml(brand)}</div>
                <div class="watch-model">${this.escapeHtml(modelName)}</div>
                <div class="watch-price">${this.escapeHtml(price)}</div>
                ${specs}
                ${variantInfo}
                ${descriptionHtml}
            </div>
        `;

        return card;
    }

    buildSpecifications(watch) {
        const specs = [];
        
        if (watch.case_size || watch.diameter) {
            specs.push(`<span class="spec-item"><i class="fas fa-ruler" aria-hidden="true"></i> ${watch.case_size || watch.diameter}</span>`);
        }
        
        if (watch.water_resistance) {
            specs.push(`<span class="spec-item"><i class="fas fa-tint" aria-hidden="true"></i> ${watch.water_resistance}m</span>`);
        }
        
        if (watch.movement) {
            specs.push(`<span class="spec-item"><i class="fas fa-cog" aria-hidden="true"></i> ${watch.movement}</span>`);
        }
        
        if (watch.case_material) {
            specs.push(`<span class="spec-item"><i class="fas fa-gem" aria-hidden="true"></i> ${watch.case_material}</span>`);
        }

        return specs.length > 0 ? `<div class="watch-specs">${specs.join('')}</div>` : '';
    }

    buildVariantInfo(watch) {
        if (!watch.has_variants || !watch.variant_count || watch.variant_count <= 1) {
            return '';
        }

        const variantText = watch.variant_count === 2 ? 'variant' : 'variants';
        const representativeText = watch.is_representative !== false ? ' (showing main variant)' : '';

        return `
            <button class="variant-info" 
                    onclick="showVariants(${watch.index})"
                    aria-label="View ${watch.variant_count - 1} more variants of this watch">
                <span>
                    <i class="fas fa-palette" aria-hidden="true"></i>
                    +${watch.variant_count - 1} more ${variantText}${representativeText}
                </span>
                <i class="fas fa-chevron-right" aria-hidden="true"></i>
            </button>
        `;
    }

    buildDescription(description) {
        if (!description || description.length === 0) {
            return '';
        }

        const shortDescription = description.length > 100 ? 
            description.substring(0, 100) + '...' : description;

        return `<div class="watch-description">${this.escapeHtml(shortDescription)}</div>`;
    }

    getPlaceholderImage() {
        return 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjI1MCIgdmlld0JveD0iMCAwIDMwMCAyNTAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIzMDAiIGhlaWdodD0iMjUwIiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik0xMzUgMTEwSDEwNVYxNDBIMTM1VjExMFoiIGZpbGw9IiM5Q0EzQUYiLz4KPHA+PHBhdGggZD0iTTE2NSAxMTBIMTk1VjE0MEgxNjVWMTEwWiIgZmlsbD0iIzlDQTNBRiIvPgo8cGF0aCBkPSJNMTA1IDE0MEgxOTVWMTcwSDEwNVYxNDBaIiBmaWxsPSIjOUNBM0FGIi8+Cjx0ZXh0IHg9IjE1MCIgeT0iMTg1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjNkI3MjgwIiBmb250LWZhbWlseT0iSW50ZXIiIGZvbnQtc2l6ZT0iMTQiPk5vIEltYWdlPC90ZXh0Pgo8L3N2Zz4K';
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    renderCard(watch) {
        if (!this.container) return;

        // Clear container
        this.container.innerHTML = '';

        if (!watch) {
            this.container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-clock" aria-hidden="true"></i>
                    <p>No watches available</p>
                </div>
            `;
            return;
        }

        const card = this.createWatchCard(watch);
        if (card) {
            this.container.appendChild(card);
        }
    }
}

// Touch/Drag Handler
class TouchHandler {
    constructor(app) {
        this.app = app;
        this.bindEvents();
    }

    bindEvents() {
        // Touch events
        document.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false });
        document.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
        document.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: false });

        // Mouse events for desktop
        document.addEventListener('mousedown', this.handleMouseDown.bind(this));
        document.addEventListener('mousemove', this.handleMouseMove.bind(this));
        document.addEventListener('mouseup', this.handleMouseUp.bind(this));

        // Keyboard events
        document.addEventListener('keydown', this.handleKeyDown.bind(this));

        // Modal click outside
        document.addEventListener('click', this.handleModalClick.bind(this));
    }

    handleTouchStart(e) {
        if (!this.isDiscoverView()) return;

        const touch = e.touches[0];
        this.startDrag(touch.clientX, touch.clientY);
    }

    handleTouchMove(e) {
        if (!this.app.state.dragState.isDragging) return;
        
        e.preventDefault();
        const touch = e.touches[0];
        this.updateDrag(touch.clientX, touch.clientY);
    }

    handleTouchEnd(e) {
        if (!this.app.state.dragState.isDragging) return;
        
        this.endDrag();
    }

    handleMouseDown(e) {
        if (!this.isDiscoverView() || !this.isWatchCard(e.target)) return;

        this.startDrag(e.clientX, e.clientY);
    }

    handleMouseMove(e) {
        if (!this.app.state.dragState.isDragging) return;
        
        this.updateDrag(e.clientX, e.clientY);
    }

    handleMouseUp(e) {
        if (!this.app.state.dragState.isDragging) return;
        
        this.endDrag();
    }

    handleKeyDown(e) {
        if (!this.isDiscoverView()) return;

        switch (e.key) {
            case 'ArrowRight':
            case 'Enter':
                e.preventDefault();
                this.app.likeWatch();
                break;
            case 'ArrowLeft':
            case ' ':
                e.preventDefault();
                this.app.passWatch();
                break;
            case 'Escape':
                e.preventDefault();
                this.app.closeModal();
                break;
        }
    }

    handleModalClick(e) {
        const modal = document.getElementById('variants-modal');
        if (modal && e.target === modal) {
            this.app.closeModal();
        }
    }

    startDrag(x, y) {
        this.app.state.dragState = {
            isDragging: true,
            startX: x,
            startY: y,
            currentX: x,
            currentY: y
        };

        const card = document.querySelector('.watch-card');
        if (card) {
            card.style.transition = 'none';
            card.style.cursor = 'grabbing';
        }
    }

    updateDrag(x, y) {
        const { dragState } = this.app.state;
        dragState.currentX = x;
        dragState.currentY = y;

        const deltaX = x - dragState.startX;
        const deltaY = y - dragState.startY;

        // Only allow horizontal dragging if movement is more horizontal than vertical
        if (Math.abs(deltaX) > Math.abs(deltaY)) {
            const card = document.querySelector('.watch-card');
            if (card) {
                const rotation = deltaX * 0.1;
                card.style.transform = `translateX(${deltaX}px) rotate(${rotation}deg)`;
                
                // Visual feedback
                if (deltaX > CONFIG.SWIPE_THRESHOLD) {
                    card.style.borderColor = 'var(--color-success)';
                } else if (deltaX < -CONFIG.SWIPE_THRESHOLD) {
                    card.style.borderColor = 'var(--color-danger)';
                } else {
                    card.style.borderColor = '';
                }
            }
        }
    }

    endDrag() {
        const { dragState } = this.app.state;
        const deltaX = dragState.currentX - dragState.startX;
        
        const card = document.querySelector('.watch-card');
        if (card) {
            card.style.transition = '';
            card.style.cursor = '';
            card.style.borderColor = '';
        }

        // Determine action based on drag distance
        if (Math.abs(deltaX) > CONFIG.DRAG_THRESHOLD) {
            if (deltaX > 0) {
                this.app.likeWatch();
            } else {
                this.app.passWatch();
            }
        } else {
            // Reset card position
            if (card) {
                card.style.transform = '';
            }
        }

        // Reset drag state
        this.app.state.dragState.isDragging = false;
    }

    isDiscoverView() {
        return this.app.ui.currentScreen === 'main-app' && 
               this.app.ui.currentView === 'discover-view';
    }

    isWatchCard(element) {
        return element.closest('.watch-card') !== null;
    }
}

// Modal Manager
class ModalManager {
    constructor() {
        this.modal = null;
    }

    showVariants(variantData) {
        this.modal = this.getOrCreateModal();
        
        const variants = variantData.variants || [];
        const representativeIndex = variantData.representative_index;

        const variantsGrid = variants.map(variant => this.createVariantCard(variant, representativeIndex)).join('');

        this.modal.innerHTML = `
            <div class="modal-content">
                <header class="modal-header">
                    <h3 id="variants-title">Watch Variants (${variants.length})</h3>
                    <button class="modal-close" onclick="closeVariantsModal()" aria-label="Close variants modal">
                        <i class="fas fa-times" aria-hidden="true"></i>
                    </button>
                </header>
                <div class="modal-body">
                    <div class="variants-grid">
                        ${variantsGrid}
                    </div>
                </div>
            </div>
        `;

        this.modal.classList.remove('hidden');
        
        // Focus management
        const firstButton = this.modal.querySelector('button');
        if (firstButton) {
            firstButton.focus();
        }
    }

    createVariantCard(variant, representativeIndex) {
        const isRepresentative = variant.index === representativeIndex;
        const representativeBadge = isRepresentative ? 
            '<span class="representative-badge">Main Variant</span>' : '';

        const imageUrl = variant.image_url || this.getPlaceholderImage();
        const modelName = variant.model_name || 'Unknown Model';
        const price = variant.price || 'Price not available';
        const description = variant.description ? 
            variant.description.substring(0, 80) + '...' : '';

        return `
            <article class="variant-card ${isRepresentative ? 'representative' : ''}" 
                     data-index="${variant.index}"
                     role="article">
                ${representativeBadge}
                <img src="${imageUrl}" 
                     alt="${modelName}" 
                     class="variant-image"
                     onerror="this.src='${this.getPlaceholderImage()}'">
                <div class="variant-info">
                    <div class="variant-name">${this.escapeHtml(modelName)}</div>
                    <div class="variant-price">${this.escapeHtml(price)}</div>
                    ${description ? `<div class="variant-description">${this.escapeHtml(description)}</div>` : ''}
                </div>
                <div class="variant-actions">
                    <button onclick="app.selectVariant(${variant.index})" 
                            class="btn btn-primary btn-sm"
                            aria-label="Select ${modelName} variant">
                        Select This Variant
                    </button>
                </div>
            </article>
        `;
    }

    getOrCreateModal() {
        let modal = document.getElementById('variants-modal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'variants-modal';
            modal.className = 'modal-overlay hidden';
            modal.setAttribute('role', 'dialog');
            modal.setAttribute('aria-labelledby', 'variants-title');
            modal.setAttribute('aria-modal', 'true');
            document.body.appendChild(modal);
        }
        return modal;
    }

    close() {
        if (this.modal) {
            this.modal.classList.add('hidden');
        }
    }

    getPlaceholderImage() {
        return 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgdmlld0JveD0iMCAwIDIwMCAxNTAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMTUwIiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik04NSA2NUg2NVY4NUg4NVY2NVoiIGZpbGw9IiM5Q0EzQUYiLz4KPHA+PHBhdGggZD0iTTEzNSA2NUgxMTVWODVIMTM1VjY1WiIgZmlsbD0iIzlDQTNBRiIvPgo8cGF0aCBkPSJNNjUgODVIMTM1Vjk1SDY1Vjg1WiIgZmlsbD0iIzlDQTNBRiIvPgo8dGV4dCB4PSIxMDAiIHk9IjExMCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzZCNzI4MCIgZm9udC1mYW1pbHk9IkludGVyIiBmb250LXNpemU9IjEyIj5ObyBJbWFnZTwvdGV4dD4KPC9zdmc+Cg==';
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Main Application Class
class WatchFinderApp {
    constructor() {
        this.state = new AppState();
        this.api = new ApiService(CONFIG.API_BASE_URL);
        this.ui = new UIManager();
        this.cardManager = new WatchCardManager(document.getElementById('watch-stack'));
        this.touchHandler = new TouchHandler(this);
        this.modalManager = new ModalManager();
        
        this.init();
    }

    async init() {
        console.log('WatchFinderApp: Starting initialization...');
        
        try {
            // Show loading screen
            this.ui.showScreen('loading-screen');
            
            // Check backend health with timeout
            await this.checkHealthWithRetry();
            
            console.log('WatchFinderApp: Health check passed');
            this.ui.showScreen('welcome-screen');
            
        } catch (error) {
            console.error('WatchFinderApp: Initialization failed:', error);
            this.ui.showError('Failed to connect to the recommendation engine. Please try again.');
        }
    }

    async checkHealthWithRetry(retries = CONFIG.MAX_RETRIES) {
        for (let i = 0; i < retries; i++) {
            try {
                await this.api.checkHealth();
                return;
            } catch (error) {
                console.warn(`Health check attempt ${i + 1} failed:`, error);
                
                if (i === retries - 1) {
                    throw error;
                }
                
                // Wait before retry
                await new Promise(resolve => setTimeout(resolve, CONFIG.RETRY_DELAY));
            }
        }
    }

    async startSession() {
        if (this.state.isLoading) return;

        this.state.isLoading = true;
        this.ui.showScreen('loading-screen');

        try {
            const data = await this.api.startSession(1);
            
            if (data.status === 'success') {
                this.state.currentWatches = data.seeds || [];
                this.state.sessionId = data.session_id;
                this.state.currentIndex = 0;
                this.state.step = 0;
                this.state.likedWatches = [];
                this.state.dislikedWatches = [];

                this.ui.showScreen('main-app');
                this.ui.showView('discover-view');
                this.cardManager.renderCard(this.state.currentWatch);
                
                this.ui.announce('Session started. Swipe or use arrow keys to rate watches.');
            } else {
                throw new Error(data.message || 'Failed to start session');
            }
        } catch (error) {
            console.error('Start session error:', error);
            this.ui.showError('Failed to start session. Please try again.');
        } finally {
            this.state.isLoading = false;
        }
    }

    async likeWatch() {
        const currentWatch = this.state.currentWatch;
        if (!currentWatch || this.state.isLoading) return;

        console.log('Liking watch:', currentWatch);
        
        this.state.addLikedWatch(currentWatch);
        
        this.ui.announce(`Liked ${currentWatch.brand} ${currentWatch.model_name}`);
        
        await this.moveToNext();
    }

    async passWatch() {
        const currentWatch = this.state.currentWatch;
        if (!currentWatch || this.state.isLoading) return;

        console.log('Passing watch:', currentWatch);
        
        this.state.addDislikedWatch(currentWatch);
        
        this.ui.announce(`Passed on ${currentWatch.brand} ${currentWatch.model_name}`);
        
        await this.moveToNext();
    }

    async moveToNext() {
        if (this.state.currentIndex < this.state.currentWatches.length - 1) {
            this.state.currentIndex++;
            this.cardManager.renderCard(this.state.currentWatch);
        } else {
            await this.getMoreRecommendations();
        }
    }

    async getMoreRecommendations() {
        if (this.state.isLoading) return;

        this.state.isLoading = true;
        this.ui.showScreen('loading-screen');

        try {
            const likedIndices = this.state.likedWatches.map(w => w.index);
            const dislikedIndices = this.state.dislikedWatches.map(w => w.index);
            const currentCandidates = this.state.currentWatches.map(w => w.index);

            const data = await this.api.getRecommendations(
                likedIndices,
                dislikedIndices,
                currentCandidates,
                this.state.step
            );

            if (data.status === 'success') {
                this.state.currentWatches = data.recommendations || [];
                this.state.step = data.step || this.state.step + 1;
                this.state.currentIndex = 0;

                this.ui.showScreen('main-app');
                this.ui.showView('discover-view');
                this.cardManager.renderCard(this.state.currentWatch);
                
                if (this.state.currentWatches.length === 0) {
                    this.ui.announce('No more recommendations available. Check your liked watches.');
                } else {
                    this.ui.announce(`Loaded ${this.state.currentWatches.length} new recommendations`);
                }
            } else {
                throw new Error(data.message || 'Failed to get recommendations');
            }
        } catch (error) {
            console.error('Get recommendations error:', error);
            this.ui.showError('Failed to get recommendations. Please try again.');
        } finally {
            this.state.isLoading = false;
        }
    }

    async showVariants(watchIndex) {
        try {
            const data = await this.api.getVariants(watchIndex);
            
            if (data.status === 'success') {
                this.modalManager.showVariants(data);
            } else {
                console.error('Failed to get variants:', data.message);
                this.ui.announce('Failed to load watch variants');
            }
        } catch (error) {
            console.error('Error getting variants:', error);
            this.ui.announce('Failed to load watch variants');
        }
    }

    selectVariant(variantIndex) {
        // This is a simplified implementation
        // In a full implementation, you might want to fetch the full watch data
        console.log('Selected variant:', variantIndex);
        this.modalManager.close();
        this.ui.announce('Variant selected');
    }

    closeModal() {
        this.modalManager.close();
    }

    // View navigation methods
    showDiscover() {
        this.state.currentView = 'discover';
        this.ui.showView('discover-view');
        this.ui.announce('Discover view');
    }

    showLiked() {
        this.state.currentView = 'liked';
        this.ui.showView('liked-view');
        this.renderLikedWatches();
        this.ui.announce(`Liked watches view. ${this.state.totalLiked} watches liked`);
    }

    showHistory() {
        this.state.currentView = 'history';
        this.ui.showView('history-view');
        this.renderHistory();
        this.ui.announce(`History view. ${this.state.totalHistory} watches reviewed`);
    }

    renderLikedWatches() {
        const grid = document.getElementById('liked-grid');
        if (!grid) return;

        if (this.state.likedWatches.length === 0) {
            grid.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-heart" aria-hidden="true"></i>
                    <p>No liked watches yet.<br>Start discovering to find watches you love!</p>
                </div>
            `;
            return;
        }

        grid.innerHTML = this.state.likedWatches.map(watch => `
            <article class="liked-watch-card" role="listitem">
                <img src="${watch.image_url}" 
                     alt="${watch.brand} ${watch.model_name}" 
                     onerror="this.src='${this.cardManager.getPlaceholderImage()}'">
                <div class="liked-watch-info">
                    <h3>${this.escapeHtml(watch.brand)}</h3>
                    <p>${this.escapeHtml(watch.model_name)}</p>
                    <p>${this.escapeHtml(watch.price)}</p>
                    <a href="${watch.product_url}" 
                       target="_blank" 
                       rel="noopener noreferrer"
                       class="btn btn-primary"
                       aria-label="View details for ${watch.brand} ${watch.model_name}">
                        View Details
                    </a>
                </div>
            </article>
        `).join('');
    }

    renderHistory() {
        const list = document.getElementById('history-list');
        if (!list) return;

        const allInteractions = [
            ...this.state.likedWatches.map(watch => ({ ...watch, action: 'liked' })),
            ...this.state.dislikedWatches.map(watch => ({ ...watch, action: 'disliked' }))
        ];

        if (allInteractions.length === 0) {
            list.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-history" aria-hidden="true"></i>
                    <p>No watch history yet.<br>Start discovering to build your preference history!</p>
                </div>
            `;
            return;
        }

        list.innerHTML = allInteractions.map(watch => `
            <article class="history-item" role="listitem">
                <img src="${watch.image_url}" 
                     alt="${watch.brand} ${watch.model_name}"
                     onerror="this.src='${this.cardManager.getPlaceholderImage()}'">
                <div class="history-item-info">
                    <h3>${this.escapeHtml(watch.brand)} ${this.escapeHtml(watch.model_name)}</h3>
                    <p>${this.escapeHtml(watch.price)}</p>
                </div>
                <div class="history-item-action ${watch.action}" 
                     aria-label="${watch.action === 'liked' ? 'Liked' : 'Disliked'}">
                    <i class="fas fa-${watch.action === 'liked' ? 'heart' : 'times'}" aria-hidden="true"></i>
                </div>
            </article>
        `).join('');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global app instance
let app;

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    try {
        console.log('DOM loaded, initializing WatchFinderApp...');
        app = new WatchFinderApp();
        console.log('WatchFinderApp initialized successfully');
    } catch (error) {
        console.error('Failed to initialize WatchFinderApp:', error);
        
        // Fallback error display
        const loadingScreen = document.getElementById('loading-screen');
        if (loadingScreen) {
            loadingScreen.innerHTML = `
                <div class="loading-content">
                    <i class="fas fa-exclamation-triangle" aria-hidden="true"></i>
                    <h2>Failed to initialize application</h2>
                    <p>Please refresh the page to try again.</p>
                    <button onclick="location.reload()" class="btn btn-primary">
                        <i class="fas fa-refresh" aria-hidden="true"></i>
                        <span>Refresh Page</span>
                    </button>
                </div>
            `;
        }
    }
});

// Global functions for HTML onclick handlers (keeping for compatibility)
function startSession() {
    if (app) {
        app.startSession();
    } else {
        console.error('App instance not available');
    }
}

function likeWatch() {
    if (app) {
        app.likeWatch();
    } else {
        console.error('App instance not available');
    }
}

function passWatch() {
    if (app) {
        app.passWatch();
    } else {
        console.error('App instance not available');
    }
}

function showVariants(watchIndex) {
    if (app) {
        app.showVariants(watchIndex);
    } else {
        console.error('App instance not available');
    }
}

function closeVariantsModal() {
    if (app) {
        app.closeModal();
    } else {
        console.error('App instance not available');
    }
}

function showDiscover() {
    if (app) {
        app.showDiscover();
    } else {
        console.error('App instance not available');
    }
}

function showLiked() {
    if (app) {
        app.showLiked();
    } else {
        console.error('App instance not available');
    }
}

function showHistory() {
    if (app) {
        app.showHistory();
    } else {
        console.error('App instance not available');
    }
} 