const slider = document.querySelector('.slider');
const items = document.querySelectorAll('.item');
const nav = document.querySelector('.nav');
let currentIndex = 0;
let autoScrollInterval;
let isTransitioning = false;

// Initialize the carousel
function initCarousel() {
  // Ensure we have items
  if (items.length === 0) {
    console.error('No carousel items found');
    return;
  }
  
  // Set initial index to show first item (WEBSEEKER)
  currentIndex = 0;
  
  // Set up the initial state
  updateCarousel();
  
  // Create indicators
  createIndicators();
  
  // Start auto-scrolling after a delay
  setTimeout(() => {
    startAutoScroll();
  }, 3000); // Give users 3 seconds to see the first slide
  
  // Add keyboard navigation
  addKeyboardNavigation();
  
  // Add intersection observer for performance
  setupIntersectionObserver();
}

// Update the carousel display with smooth transitions
function updateCarousel() {
  // Prevent multiple simultaneous transitions
  if (isTransitioning) return;
  isTransitioning = true;
  
  // Remove all classes first
  items.forEach(item => {
    item.classList.remove('active', 'next', 'prev');
  });
  
  // Calculate indices with proper wrapping
  const prevIndex = (currentIndex - 1 + items.length) % items.length;
  const nextIndex = (currentIndex + 1) % items.length;
  
  // Apply classes with a slight delay for smoother animation
  requestAnimationFrame(() => {
    items[currentIndex].classList.add('active');
    items[nextIndex].classList.add('next');
    items[prevIndex].classList.add('prev');
  });
  
  // Update indicators
  updateIndicators();
  
  // Reset transition lock after animation completes
  setTimeout(() => {
    isTransitioning = false;
  }, 700); // Match CSS transition duration
}

// Handle navigation
function navigate(direction) {
  // Prevent navigation during transition
  if (isTransitioning) return;
  
  // Reset auto-scroll timer
  resetAutoScroll();
  
  if (direction === 'next') {
    currentIndex = (currentIndex + 1) % items.length;
  } else if (direction === 'prev') {
    currentIndex = (currentIndex - 1 + items.length) % items.length;
  } else if (!isNaN(direction)) {
    // Direct navigation to specific slide
    const targetIndex = parseInt(direction) % items.length;
    if (targetIndex === currentIndex) return; // Already on this slide
    currentIndex = targetIndex;
  }
  
  updateCarousel();
}

// Create indicator dots with labels
function createIndicators() {
  const indicatorsContainer = document.createElement('div');
  indicatorsContainer.className = 'indicators';
  
  const labels = [
    'WebSeeker', 'PortScanner', 'Site Index', 'File Fender', 
    'InfoCrypt', 'SnapSpeak AI', 'TrueShot AI', 'InfoSight AI', 'LANA AI', 
    'CyberSentry AI', 'Inkwell AI', 'Tracklyst'
  ];
  
  items.forEach((item, i) => {
    const indicator = document.createElement('button');
    indicator.className = 'indicator';
    indicator.setAttribute('data-index', i);
    indicator.setAttribute('aria-label', `Go to ${labels[i] || 'slide ' + (i + 1)}`);
    indicator.setAttribute('title', labels[i] || `Slide ${i + 1}`);
    
    indicator.addEventListener('click', function() {
      if (!isTransitioning) {
        navigate(this.getAttribute('data-index'));
      }
    });
    
    indicatorsContainer.appendChild(indicator);
  });
  
  document.querySelector('main').appendChild(indicatorsContainer);
}

// Update indicator dots
function updateIndicators() {
  const indicators = document.querySelectorAll('.indicator');
  indicators.forEach((indicator, index) => {
    if (index === currentIndex) {
      indicator.classList.add('active');
      indicator.setAttribute('aria-current', 'true');
    } else {
      indicator.classList.remove('active');
      indicator.setAttribute('aria-current', 'false');
    }
  });
}

// Auto-scroll functionality
function startAutoScroll() {
  // Clear any existing interval
  if (autoScrollInterval) {
    clearInterval(autoScrollInterval);
  }
  
  autoScrollInterval = setInterval(() => {
    if (!isTransitioning && !document.hidden) {
      navigate('next');
    }
  }, 6000); // Change slide every 6 seconds
}

function resetAutoScroll() {
  clearInterval(autoScrollInterval);
  startAutoScroll();
}

function stopAutoScroll() {
  if (autoScrollInterval) {
    clearInterval(autoScrollInterval);
  }
}

// Keyboard navigation
function addKeyboardNavigation() {
  document.addEventListener('keydown', (e) => {
    if (isTransitioning) return;
    
    switch(e.key) {
      case 'ArrowLeft':
        e.preventDefault();
        navigate('prev');
        break;
      case 'ArrowRight':
        e.preventDefault();
        navigate('next');
        break;
      case 'Home':
        e.preventDefault();
        navigate(0);
        break;
      case 'End':
        e.preventDefault();
        navigate(items.length - 1);
        break;
      case ' ': // Spacebar
        e.preventDefault();
        if (autoScrollInterval) {
          stopAutoScroll();
        } else {
          startAutoScroll();
        }
        break;
    }
  });
}

// Setup intersection observer for performance
function setupIntersectionObserver() {
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) {
          stopAutoScroll();
        } else {
          startAutoScroll();
        }
      });
    }, { threshold: 0.5 });
    
    observer.observe(slider);
  }
}

// Event listeners for navigation buttons
const prevBtn = document.querySelector('.btn.prev');
const nextBtn = document.querySelector('.btn.next');

if (prevBtn && nextBtn) {
  prevBtn.addEventListener('click', () => {
    if (!isTransitioning) {
      navigate('prev');
    }
  });
  
  nextBtn.addEventListener('click', () => {
    if (!isTransitioning) {
      navigate('next');
    }
  });
  
  // Add visual feedback
  [prevBtn, nextBtn].forEach(btn => {
    btn.addEventListener('mousedown', () => {
      btn.style.transform = 'scale(0.95)';
    });
    
    btn.addEventListener('mouseup', () => {
      btn.style.transform = '';
    });
  });
}

// Pause auto-scroll on hover over main content
if (slider) {
  slider.addEventListener('mouseenter', stopAutoScroll);
  slider.addEventListener('mouseleave', () => {
    // Only restart if user isn't interacting
    setTimeout(() => {
      if (!isTransitioning) {
        startAutoScroll();
      }
    }, 1000);
  });
}

// Handle visibility change (pause when tab is not visible)
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    stopAutoScroll();
  } else {
    startAutoScroll();
  }
});

// Touch/swipe support for mobile
let touchStartX = 0;
let touchEndX = 0;
let touchStartY = 0;
let touchEndY = 0;

if (slider) {
  slider.addEventListener('touchstart', (e) => {
    touchStartX = e.changedTouches[0].screenX;
    touchStartY = e.changedTouches[0].screenY;
    stopAutoScroll();
  }, { passive: true });
  
  slider.addEventListener('touchend', (e) => {
    touchEndX = e.changedTouches[0].screenX;
    touchEndY = e.changedTouches[0].screenY;
    handleSwipe();
  }, { passive: true });
}

function handleSwipe() {
  const swipeThreshold = 50;
  const diffX = touchStartX - touchEndX;
  const diffY = touchStartY - touchEndY;
  
  // Only trigger if horizontal swipe is more significant than vertical
  if (Math.abs(diffX) > Math.abs(diffY) && Math.abs(diffX) > swipeThreshold && !isTransitioning) {
    if (diffX > 0) {
      // Swiped left - go to next
      navigate('next');
    } else {
      // Swiped right - go to previous
      navigate('prev');
    }
  }
  
  // Restart auto-scroll after swipe
  setTimeout(() => {
    startAutoScroll();
  }, 1000);
}

// Handle window resize with debouncing
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    updateCarousel();
  }, 250);
});

// Preload next images for smoother transitions
function preloadImages() {
  const nextIndex = (currentIndex + 1) % items.length;
  const prevIndex = (currentIndex - 1 + items.length) % items.length;
  
  [nextIndex, prevIndex].forEach(index => {
    const item = items[index];
    const bgImage = window.getComputedStyle(item).backgroundImage;
    const url = bgImage.slice(4, -1).replace(/"/g, '');
    
    if (url && url !== 'none') {
      const img = new Image();
      img.src = url;
    }
  });
}

// Initialize on page load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initCarousel);
} else {
  // DOMContentLoaded has already fired
  initCarousel();
}

// Preload images periodically
setInterval(preloadImages, 5000);

// Performance monitoring (development only)
if (window.location.hostname === 'localhost') {
  let frameCount = 0;
  let lastTime = performance.now();
  
  function measureFPS() {
    frameCount++;
    const currentTime = performance.now();
    
    if (currentTime >= lastTime + 1000) {
      const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
      // FPS logging removed for production (uncomment for debugging)
      // console.log(`FPS: ${fps}`);
      frameCount = 0;
      lastTime = currentTime;
    }
    
    requestAnimationFrame(measureFPS);
  }
  
  // measureFPS(); // Uncomment to monitor FPS
}