import { TestBed } from '@angular/core/testing';

import { SendFileDataService } from './send-file-data.service';

describe('SendFileDataService', () => {
  let service: SendFileDataService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(SendFileDataService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
