import { Component, ChangeDetectorRef } from '@angular/core'; // <--- 1. Import ChangeDetectorRef
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AuthService } from '../../../services/auth.service';

@Component({
  selector: 'app-partners-predictions',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './partners-predictions.html', // Fixed file extension (.html)
  styleUrls: ['./partners-predictions.css']   // Fixed file extension (.css)
})
export class PartnersPredictionsComponent {
  
  selectedDisease: string = '';
  isLoading: boolean = false;
  
  // Results
  predictionResult: number | null = null;
  predictionImage: string | null = null;
  errorMessage: string = '';

  commonDiseases: string[] = [
    'Influenza', 'COVID-19', 'Common Cold', 'Gastroenteritis', 
    'Migraine', 'Bronchitis', 'Pneumonia', 'Diabetes Type 2'
  ];

  constructor(
    private authService: AuthService,
    private cdr: ChangeDetectorRef // <--- 2. Inject it here
  ) {}

  generatePrediction() {
    if (!this.selectedDisease.trim()) return;

    this.isLoading = true;
    this.errorMessage = '';
    this.predictionResult = null;
    this.predictionImage = null;

    this.authService.getPrediction(this.selectedDisease).subscribe({
      next: (res: any) => {
        this.isLoading = false;
        
        if (res.success) {
          this.predictionResult = res.prediction;
          // The backend sends the "data:image/png;base64," prefix, so we just assign it
          this.predictionImage = res.plot_image; 
        }
        
        // <--- 3. FORCE SCREEN UPDATE
        this.cdr.detectChanges(); 
      },
      error: (err) => {
        this.isLoading = false;
        console.error("Prediction Error:", err);
        
        // Handle nicely if Python fails or no data found
        if (err.error && err.error.error) {
            this.errorMessage = err.error.error;
        } else {
            this.errorMessage = "Failed to generate prediction. Please try again.";
        }
        
        // <--- 3. FORCE SCREEN UPDATE ON ERROR
        this.cdr.detectChanges();
      }
    });
  }
}