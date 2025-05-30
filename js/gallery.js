document.addEventListener('DOMContentLoaded', function() {
    console.log('Gallery script starting...');
    const gallery = document.querySelector('.horizontal-image-gallery');
    if (!gallery) {
        console.error('Gallery element (.horizontal-image-gallery) not found!');
        return;
    }
    console.log('Gallery element:', gallery);

    const track = gallery.querySelector('.gallery-track');
    const items = Array.from(gallery.querySelectorAll('.gallery-item')); // Changed to query from gallery, more specific
    const prevButton = gallery.querySelector('.prev-arrow');
    const nextButton = gallery.querySelector('.next-arrow');
    const viewport = gallery.querySelector('.gallery-viewport');

    console.log('Track:', track);
    console.log('Items count:', items.length);
    console.log('Prev button:', prevButton);
    console.log('Next button:', nextButton);
    console.log('Viewport:', viewport);

    if (!track || !prevButton || !nextButton || !viewport || items.length === 0) {
        console.warn('Essential gallery components missing or no items found. Hiding arrows or exiting.');
        if (prevButton) prevButton.style.display = 'none';
        if (nextButton) nextButton.style.display = 'none';
        // If critical elements are missing, disable the script further
        if (!track || !viewport || items.length === 0) {
            console.error("Cannot proceed without track, viewport, or items.");
            return;
        }
    }

    let itemWidth = 0; // Width of a single item including its margin-right
    let itemsPerScreen = 0; // How many items can fit in the viewport
    let currentOffset = 0; // Current translateX value for the track
    let maxOffset = 0; // Maximum scrollable distance

    function calculateDimensions() {
        console.log('Calculating dimensions...');
        if (items.length === 0) {
            console.log('No items to calculate dimensions for.');
            if (prevButton) prevButton.style.display = 'none';
            if (nextButton) nextButton.style.display = 'none';
            return;
        }

        // Ensure the first item exists before trying to get its style and offsetWidth
        const firstItem = items[0];
        if (!firstItem) {
            console.error("First item not found in 'items' array.");
            return;
        }
        const firstItemStyle = window.getComputedStyle(firstItem);
        const marginRight = parseFloat(firstItemStyle.marginRight);
        if (isNaN(marginRight)) {
            console.error("Could not parse marginRight of the first item. CSS might be missing or incorrect for .gallery-item.");
            return; // Stop if margin is not a number
        }
        itemWidth = firstItem.offsetWidth + marginRight;
        console.log(`Calculated itemWidth: ${itemWidth} (offsetWidth: ${firstItem.offsetWidth} + marginRight: ${marginRight})`);
        
        const viewportWidth = viewport.clientWidth;
        console.log('Viewport clientWidth:', viewportWidth);

        if (itemWidth <= 0) {
            console.error('itemWidth is zero or negative. Cannot calculate itemsPerScreen or maxOffset correctly. Check CSS for .gallery-item width.');
            if (prevButton) { prevButton.disabled = true; prevButton.style.opacity = "0.5"; } // Visually indicate disabled
            if (nextButton) { nextButton.disabled = true; nextButton.style.opacity = "0.5"; }
            return;
        }
        
        itemsPerScreen = Math.floor(viewportWidth / itemWidth);
        console.log('Items per screen (approx):', itemsPerScreen);

        let trackContentWidth = 0;
        items.forEach((item, index) => {
            const style = window.getComputedStyle(item);
            trackContentWidth += item.offsetWidth;
            if (index < items.length - 1) { // Add margin for all but the last item
                 const itemMarginRight = parseFloat(style.marginRight);
                 if (!isNaN(itemMarginRight)) {
                    trackContentWidth += itemMarginRight;
                 } else {
                    console.warn(`Could not parse margin-right for item at index ${index}`);
                 }
            }
        });
        console.log('Total track content width:', trackContentWidth);
        
        maxOffset = Math.max(0, trackContentWidth - viewportWidth);
        console.log('Max offset (scrollable distance):', maxOffset);
        
        // If currentOffset is beyond the new maxOffset (e.g., after resize shrinking viewport), adjust it
        if (currentOffset > maxOffset) {
            console.log(`Current offset (${currentOffset}) exceeds new maxOffset (${maxOffset}). Adjusting.`);
            currentOffset = maxOffset;
            track.style.transform = `translateX(-${currentOffset}px)`;
        }
        
        updateButtonStates();
    }

    function slide(direction) {
        console.log(`Sliding ${direction}. Current offset before slide: ${currentOffset}`);
        const scrollAmount = itemWidth; // Scroll by one item's width
        console.log('Scroll amount (itemWidth):', scrollAmount);

        if (scrollAmount <= 0) {
            console.error("Scroll amount (itemWidth) is zero or negative. Cannot slide. Check item dimensions.");
            return;
        }

        let targetOffset = currentOffset;
        if (direction === 'next') {
            targetOffset += scrollAmount;
        } else if (direction === 'prev') {
            targetOffset -= scrollAmount;
        }

        // Clamp targetOffset to be within [0, maxOffset]
        currentOffset = Math.max(0, Math.min(targetOffset, maxOffset));
        
        console.log('New calculated offset:', currentOffset);
        track.style.transform = `translateX(-${currentOffset}px)`;
        updateButtonStates();
    }

    function updateButtonStates() {
        if (!prevButton || !nextButton) {
            console.warn("Buttons not available for state update.");
            return;
        }
        console.log(`Updating button states. Current offset: ${currentOffset}, Max offset: ${maxOffset}, Items per screen: ${itemsPerScreen}, Total items: ${items.length}`);

        // Disable prev if at the beginning
        prevButton.disabled = currentOffset <= 0;

        // Disable next if at the end (or very close to it, using a small epsilon for float comparisons)
        const epsilon = 1; 
        nextButton.disabled = currentOffset >= maxOffset - epsilon;

        // If all items fit on screen, or not enough items to scroll, disable both
        // This condition `items.length <= itemsPerScreen` is tricky if itemsPerScreen is 0 (e.g. itemWidth > viewportWidth)
        // A more robust check is simply if maxOffset is 0.
        if (maxOffset <= 0) {
            console.log('Max offset is 0 or less. Not enough content to scroll, or all items fit. Disabling both buttons.');
            prevButton.disabled = true;
            nextButton.disabled = true;
        }
        
        console.log(`Prev button disabled: ${prevButton.disabled}, Next button disabled: ${nextButton.disabled}`);
        // Optionally, add visual cues for disabled state beyond the default browser style
        prevButton.style.opacity = prevButton.disabled ? "0.5" : "1";
        nextButton.style.opacity = nextButton.disabled ? "0.5" : "1";
    }

    // Attach event listeners only if buttons exist
    if (nextButton) {
        nextButton.addEventListener('click', () => slide('next'));
    } else {
        console.warn("Next button not found, cannot attach click listener.");
    }
    if (prevButton) {
        prevButton.addEventListener('click', () => slide('prev'));
    } else {
        console.warn("Previous button not found, cannot attach click listener.");
    }
    

    calculateDimensions(); // Initial calculation

    let resizeTimeout;
    window.addEventListener('resize', () => {
        console.log('Window resize detected.');
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            console.log('Debounced resize: recalculating dimensions and position.');
            const oldMaxOffset = maxOffset;
            const oldCurrentOffset = currentOffset;
            
            calculateDimensions(); // This will update itemWidth, maxOffset, itemsPerScreen and call updateButtonStates

            // Try to maintain scroll position proportionally if possible
            if (maxOffset <= 0) { // If no scrollable area anymore
                currentOffset = 0;
            } else if (oldMaxOffset > 0) { // If there was a scrollable area before
                const scrollRatio = oldCurrentOffset / oldMaxOffset;
                currentOffset = Math.round(maxOffset * scrollRatio);
            } else { // Was not scrollable, now is (or still not, covered by maxOffset <= 0)
                currentOffset = 0; // Reset to start
            }
            
            // Ensure currentOffset is clamped after proportional adjustment
            currentOffset = Math.max(0, Math.min(currentOffset, maxOffset));
            
            console.log('Resize: Adjusted currentOffset:', currentOffset);
            track.style.transform = `translateX(-${currentOffset}px)`;
            // updateButtonStates is already called within calculateDimensions, but call again if offset was adjusted here.
            updateButtonStates(); 
        }, 250);
    });

    // Accessibility: Focus handling (simplified)
    items.forEach(item => {
        item.addEventListener('focus', () => {
            // This is a simplified focus handling. A robust solution would be more complex.
            // For now, ensure focused items are scrolled into view if they gain focus by keyboard.
            const itemRect = item.getBoundingClientRect();
            const viewportRect = viewport.getBoundingClientRect();

            let requiredScroll = currentOffset;
            if (itemRect.left < viewportRect.left) { // Item is to the left of viewport
                requiredScroll -= (viewportRect.left - itemRect.left);
            } else if (itemRect.right > viewportRect.right) { // Item is to the right of viewport
                requiredScroll += (itemRect.right - viewportRect.right);
            } else {
                return; // Already in view
            }
            
            currentOffset = Math.max(0, Math.min(requiredScroll, maxOffset));
            track.style.transform = `translateX(-${currentOffset}px)`;
            updateButtonStates();
        });
    });
    console.log('Gallery script initialized.');
});
