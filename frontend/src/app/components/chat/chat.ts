import { Component, ElementRef, ViewChild, AfterViewChecked } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';

interface ChatMessage {
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.html',
  styleUrls: ['./chat.css']
})
export class ChatComponent implements AfterViewChecked {
  @ViewChild('scrollContainer') private scrollContainer!: ElementRef;
  
  newMessage: string = '';
  isDropdownOpen: boolean = false;
  isTyping: boolean = false; // To show "AI is typing..."

  // Initial Message
  messages: ChatMessage[] = [
    { text: 'Hello! I am your AI health assistant. How can I help you today?', sender: 'bot', timestamp: new Date() }
  ];

  constructor(private authService: AuthService, private router: Router) {}

  // Auto-scroll to bottom whenever a new message is added
  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  scrollToBottom(): void {
    try {
      this.scrollContainer.nativeElement.scrollTop = this.scrollContainer.nativeElement.scrollHeight;
    } catch(err) { }
  }

  toggleDropdown() {
    this.isDropdownOpen = !this.isDropdownOpen;
  }

  sendMessage() {
    if (this.newMessage.trim() === '') return;

    // 1. Add User Message
    this.messages.push({
      text: this.newMessage,
      sender: 'user',
      timestamp: new Date()
    });

    const userQuery = this.newMessage; // Store for API call later
    this.newMessage = ''; // Clear input
    this.isTyping = true; // Show loader

    // 2. Simulate AI API Call (Replace this setTimeout with real HTTP call later)
    setTimeout(() => {
      this.messages.push({
        text: `I received your message: "${userQuery}". This is a placeholder for the AI response.`,
        sender: 'bot',
        timestamp: new Date()
      });
      this.isTyping = false;
    }, 1500); // Faking a 1.5 second delay
  }

  goToProfile() {
    this.router.navigate(['/profile']); // You will need to create this route later
  }

  logout() {
    this.authService.logout();
  }
}