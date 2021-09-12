import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from 'src/environments/environment';

@Injectable({
  providedIn: 'root'
})
export class SendFileDataService {

  constructor(private http: HttpClient) { }

  sendPath(path: string) {
    return this.http.get(environment.url + "GetFilePath/" + encodeURIComponent(path), { responseType: 'text' });
  }
}
