import { Component, OnInit } from '@angular/core';
import { SendFileDataService } from 'src/app/services/send-file-data.service';


@Component({
  selector: 'app-uploade-file',
  templateUrl: './uploade-file.component.html',
  styleUrls: ['./uploade-file.component.css']
})
export class UploadeFileComponent implements OnInit {
  path: string
  constructor(private sendService: SendFileDataService) { }

  ngOnInit(): void {
  }
  upload(event: Event) {
    console.log(event);
    this.path = (<HTMLInputElement>event.target).value;
    console.log(this.path);
    this.sendService.sendPath(this.path).subscribe(res => {
      console.log("succses to connect " + res);
    }, err => { debugger; console.log("err") });
  }
}
