import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from 'src/environments/environment';

@Injectable({
  providedIn: 'root'
})
export class SendFileDataService {

  constructor(private http: HttpClient) { }

  sendPath(path: string) {
    // return this.http.post(environment.url + "GetFilePath", path);
    // return this.http.get(environment.url + "GetFilePath/" + encodeURIComponent(path));
    return this.http.get(environment.url + "GetFilePath/" + "jhkh", { responseType: 'text' });
    //   return this.http.post(
    //     'http://10.0.1.19/login',
    //     {email, password},
    //     {responseType: 'text'})
  }
}
