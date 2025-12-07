import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common'; 
import { RouterLink } from '@angular/router';
import { AuthService } from '../../services/auth.service';

interface DiagnosticDetails {
  most_probable_diagnostic: string;
  second_most_probable: string;
  third_most_probable: string;
}

interface MedicalReport {
  report_date: string;
  final_diagnostic: DiagnosticDetails;
  recommendation: string;
}

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule, RouterLink], 
  templateUrl: './history.html',
  styleUrls: ['./history.css']
})
export class HistoryComponent implements OnInit {
  
  reports: MedicalReport[] = [];
  isLoading: boolean = true;

  constructor(
    private authService: AuthService,
    private cdr: ChangeDetectorRef // <--- 1. INJECT THIS
  ) {}

  ngOnInit() {
    this.fetchHistory();
  }

  fetchHistory() {
    console.log("🚀 Fetching history...");

    this.authService.getProfile().subscribe({
      next: (userProfile: any) => {
        console.log("✅ History Data Received");
        
        const rawHistory = userProfile.previous_conversations || [];
        
        // Sort by date: Newest first
        this.reports = rawHistory.sort((a: MedicalReport, b: MedicalReport) => {
          return new Date(b.report_date).getTime() - new Date(a.report_date).getTime();
        });

        this.isLoading = false;
        
        // <--- 2. FORCE SCREEN UPDATE
        this.cdr.detectChanges(); 
      },
      error: (err) => {
        console.error("❌ Failed to load history", err);
        this.isLoading = false;
        
        // <--- 3. FORCE SCREEN UPDATE ON ERROR
        this.cdr.detectChanges();
      }
    });
  }
}