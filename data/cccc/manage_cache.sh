#!/bin/bash

# Script to manage Hugging Face cache migration and cleanup

echo "===== Hugging Face Cache Management Script ====="
echo ""
echo "Current cache usage:"
echo "-------------------"
df -h / | grep -E "Filesystem|/dev/vda1"
echo ""
echo "Old cache location: ~/.cache/huggingface/"
if [ -d ~/.cache/huggingface/ ]; then
    du -sh ~/.cache/huggingface/
else
    echo "Old cache not found"
fi
echo ""
echo "New cache location: /mnt/local/huggingface_cache/"
if [ -d /mnt/local/huggingface_cache/ ]; then
    du -sh /mnt/local/huggingface_cache/
else
    echo "New cache not found"
fi
echo ""
echo "Available space on /mnt/local:"
df -h /mnt/local | grep -E "Filesystem|/dev/vdc"
echo ""

echo "Options:"
echo "1) Move existing cache to /mnt/local (preserves downloaded data)"
echo "2) Delete old cache to free up space (will re-download data)"
echo "3) Show detailed cache contents"
echo "4) Exit without changes"
echo ""
read -p "Select option (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Moving cache to /mnt/local..."
        echo "This may take a while depending on cache size..."
        
        # Create destination if it doesn't exist
        mkdir -p /mnt/local/huggingface_cache
        
        # Move cache contents
        if [ -d ~/.cache/huggingface/ ]; then
            rsync -av --progress ~/.cache/huggingface/ /mnt/local/huggingface_cache/
            
            echo ""
            read -p "Move completed. Delete old cache to free up space? (y/n): " delete_old
            if [ "$delete_old" = "y" ]; then
                rm -rf ~/.cache/huggingface/
                echo "Old cache deleted."
            else
                echo "Old cache preserved. You can delete it later with: rm -rf ~/.cache/huggingface/"
            fi
        else
            echo "No old cache found to move."
        fi
        ;;
        
    2)
        echo ""
        read -p "Are you sure you want to delete the old cache? This will free ~209GB but data will need to be re-downloaded (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            rm -rf ~/.cache/huggingface/
            echo "Old cache deleted."
            echo ""
            echo "New space available:"
            df -h / | grep -E "Filesystem|/dev/vda1"
        else
            echo "Deletion cancelled."
        fi
        ;;
        
    3)
        echo ""
        echo "Cache contents:"
        echo "---------------"
        if [ -d ~/.cache/huggingface/ ]; then
            echo "Old cache (~/.cache/huggingface/):"
            du -sh ~/.cache/huggingface/*/ 2>/dev/null | head -20
            echo ""
        fi
        
        if [ -d /mnt/local/huggingface_cache/ ]; then
            echo "New cache (/mnt/local/huggingface_cache/):"
            du -sh /mnt/local/huggingface_cache/*/ 2>/dev/null | head -20
        fi
        ;;
        
    4)
        echo "Exiting without changes."
        ;;
        
    *)
        echo "Invalid option. Exiting."
        ;;
esac

echo ""
echo "===== Done ====="
