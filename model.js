let messages = [];
const messagesContainer = document.getElementById('messages');
const inputElement = document.getElementById('input');
const sendButton = document.getElementById('send-btn');
const fileInputElement = document.getElementById('fileInput');

// Function to update the UI with messages
function updateMessages() {
    messagesContainer.innerHTML = messages.map((message) => {
        return `<div class="${message.type}"><div class="message">${message.content}</div></div>`;
    }).join('');
    messagesContainer.scrollTop = messagesContainer.scrollHeight; // Scroll to the bottom
}

// Function to handle message send
async function sendMessage() {
    const inputValue = inputElement.value.trim();
    if (!inputValue) return;

    messages.push({ type: 'user', content: inputValue });
    inputElement.value = ''; // Clear input field
    updateMessages(); // Update the UI to reflect new message

    try {
        const response = await fetch('http://127.0.0.1:8080/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `query=${encodeURIComponent(inputValue)}`,
        });
        const data = await response.json();
        messages.push({ type: 'bot', content: data.answer });
    } catch (error) {
        messages.push({ type: 'bot', content: 'Failed to get response.' });
    } finally {
        updateMessages(); // Update the UI to reflect bot response
    }
}

// Event listener for the "Send" button
sendButton.addEventListener('click', sendMessage);

// Event listener for the "Enter" key
inputElement.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevent default behavior (like adding a newline)
        sendMessage(); // Call sendMessage function
    }
});

// Function to handle file upload
fileInputElement.addEventListener('change', async (event) => {
    const files = event.target.files;
    if (!files.length) return;

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    try {
        const response = await fetch('http://127.0.0.1:8080/upload', {
            method: 'POST',
            body: formData,
        });
        if (response.ok) {
            // Add success message to the messages array
            const successMessage = { type: 'bot', content: 'Files uploaded successfully.' };
            messages.push(successMessage);
            updateMessages(); // Update the UI to show the success message

            // Remove the message after 10 seconds (10000 milliseconds)
            setTimeout(() => {
                const index = messages.indexOf(successMessage);
                if (index > -1) {
                    messages.splice(index, 1); // Remove the message from the array
                    updateMessages(); // Update the UI to remove the message
                }
            }, 10000);
        } else {
            messages.push({ type: 'bot', content: 'File upload failed.' });
        }
    } catch (error) {
        messages.push({ type: 'bot', content: 'File upload failed.' });
    } finally {
        updateMessages(); // Update the UI after handling file upload
    }
});


// Function to convert messages to CSV format
function convertToCSV(messages) {
    const header = 'User,Message\n';
    const rows = messages.map(message => {
        return `${message.type === 'user' ? 'User' : 'Bot'},${message.content.replace(/,/g, ' ')}\n`;
    }).join('');
    return header + rows;
}

// Function to trigger CSV download
function downloadCSV(csvContent) {
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('href', url);
    a.setAttribute('download', 'conversation.csv');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// Event listener for the "Share" button
document.getElementById('share-btn').addEventListener('click', () => {
    const csvContent = convertToCSV(messages);
    downloadCSV(csvContent);
});

// Event listener for the "View Conversation" button
document.getElementById('conversation-btn').addEventListener('click', () => {
    alert(messages.map(msg => `${msg.type === 'user' ? 'User' : 'Bot'}: ${msg.content}`).join('\n'));
});
