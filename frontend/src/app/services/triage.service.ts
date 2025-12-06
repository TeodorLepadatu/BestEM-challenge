import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class TriageService {
  private baseUrl = 'http://127.0.0.1:8000'; 

  constructor(private http: HttpClient) { }

  // Get list of all chats for the sidebar
  getConversations(): Observable<any[]> {
    return this.http.get<any[]>(`${this.baseUrl}/conversations`);
  }

  // Load history of a specific chat
  getConversationDetails(id: string): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/conversations/${id}`);
  }

  // UPDATED: Note that this now points to "/chat_step", NOT "/triage_step"
  sendMessage(message: string, conversationId: string | null): Observable<any> {
    const payload = { 
      message: message, 
      conversation_id: conversationId 
    };
    return this.http.post<any>(`${this.baseUrl}/chat_step`, payload);
  }
}