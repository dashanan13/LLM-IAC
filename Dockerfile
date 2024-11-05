# Use the official Nginx image from Docker Hub
FROM nginx:alpine

# Copy the HTML file into the container's default web directory
COPY index.html /usr/share/nginx/html/

# Expose port 80 to access the web server
EXPOSE 80
