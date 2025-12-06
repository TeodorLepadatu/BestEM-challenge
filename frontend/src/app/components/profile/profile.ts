import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { Component, OnInit, ChangeDetectorRef } from '@angular/core'; // <-- Import ChangeDetectorRef here

@Component({
  selector: 'app-profile',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './profile.html',
  styleUrls: ['./profile.css']
})
export class ProfileComponent implements OnInit {
  user: any = null;
  isLoading: boolean = true;
  errorMessage: string = '';

  constructor(private authService: AuthService, private router: Router, private cdr: ChangeDetectorRef) {}

  ngOnInit() {
    this.fetchProfile();
  }

  fetchProfile() {
    console.log("🟡 Frontend: Starting profile fetch..."); // DEBUG LOG

    this.authService.getProfile().subscribe({
      next: (data) => {
        console.log("🟢 Frontend: Data received!", data); // DEBUG LOG
        this.user = data;
        this.isLoading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error("🔴 Frontend: Error received!", err); // DEBUG LOG
        
        // Check actual error details in the browser console
        if (err.status === 200) {
            console.error("⚠️ Error is 200 OK? This means JSON parsing failed.");
        }

        this.errorMessage = 'Failed to load profile data.';
        this.isLoading = false;
        
        if (err.status === 401 || err.status === 403) {
          this.authService.logout();
        }
      }
    });
  }

  goBack() {
    this.router.navigate(['/chat']);
  }

  logout() {
    this.authService.logout();
  }
}