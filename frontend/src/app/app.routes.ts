import { Routes } from '@angular/router';
import { HomeComponent } from './components/home/home'; 
import { RegisterComponent } from './components/register/register'; 
import { LoginComponent } from './components/login/login';
import { ChatComponent } from './components/chat/chat';
import { authGuard } from './guards/auth.guard';
import { ProfileComponent } from './components/profile/profile';

export const routes: Routes = [
    
    {path: '', redirectTo: 'home', pathMatch: 'full' },
    {path: 'home', component: HomeComponent },
    {path: 'register', component: RegisterComponent},
    {path: 'login', component: LoginComponent},
    {path: 'chat', component: ChatComponent, canActivate: [authGuard]},
    {path: 'profile', component: ProfileComponent, canActivate: [authGuard] },


];